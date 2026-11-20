// MIT License

// Copyright (c) 2023 Andy Pan

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

package ants_test

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/panjf2000/ants/v2"
)

// blockUntil returns a task that blocks until the returned channel is closed,
// giving tests precise control over in-flight leased tasks.
func blockUntil(t *testing.T) (func(), chan struct{}) {
	t.Helper()
	release := make(chan struct{})
	return func() {
		<-release
	}, release
}

// TestTenantFairAffinity verifies that one tenant stays on the same pool while
// its lease generation is alive and that the affinity is refreshed per task.
func TestTenantFairAffinity(t *testing.T) {
	mp, err := ants.NewMultiPool(4, 4, ants.TenantFair, ants.WithNonblocking(true))
	require.NoError(t, err)
	defer mp.ReleaseTimeout(3 * time.Second)

	run, release := blockUntil(t)
	const tenant = "affine-tenant"
	for i := 0; i < 4; i++ {
		require.NoError(t, mp.SubmitTenant(tenant, run))
	}

	idx, err := mp.TenantPool(tenant)
	require.NoError(t, err)
	require.GreaterOrEqual(t, idx, 0)
	leases, err := mp.TenantLeases(tenant)
	require.NoError(t, err)
	require.Equal(t, 4, leases)

	// All four tasks must have been placed on the affinity pool.
	running, err := mp.RunningByIndex(idx)
	require.NoError(t, err)
	require.Equal(t, 4, running)

	stats, err := mp.LeaseStats()
	require.NoError(t, err)
	require.EqualValues(t, 4, stats.Acquired)
	require.EqualValues(t, 3, stats.Renewed)
	require.Zero(t, stats.Migrated)
	require.Zero(t, stats.Expired)
	require.EqualValues(t, 4, stats.Dispatched)
	require.Zero(t, stats.Completed)

	close(release)
	require.Eventually(t, func() bool {
		s, _ := mp.LeaseStats()
		return s.Completed == 4 && s.Released == 4
	}, 3*time.Second, 5*time.Millisecond)

	leases, err = mp.TenantLeases(tenant)
	require.NoError(t, err)
	require.Zero(t, leases)
}

// TestTenantFairQuota verifies a single tenant can never occupy the whole
// multi-pool: at least one pool's worth of capacity stays available for the
// other tenants.
func TestTenantFairQuota(t *testing.T) {
	const poolCount, capPerPool = 4, 4
	mp, err := ants.NewMultiPool(poolCount, capPerPool, ants.TenantFair,
		ants.WithNonblocking(true), ants.WithTenantLeaseTTL(time.Minute))
	require.NoError(t, err)
	defer mp.ReleaseTimeout(3 * time.Second)

	run, release := blockUntil(t)
	// Tenant A may hold at most (poolCount-1)*capPerPool = 12 leases.
	for i := 0; i < (poolCount-1)*capPerPool; i++ {
		require.NoError(t, mp.SubmitTenant("A", run))
	}
	// The next submission is rejected: the quota keeps one whole pool for
	// the other tenants.
	require.ErrorIs(t, mp.SubmitTenant("A", run), ants.ErrPoolOverload)

	// The reserved pool is fully available to tenant B.
	for i := 0; i < capPerPool; i++ {
		require.NoError(t, mp.SubmitTenant("B", run))
	}
	leasesB, err := mp.TenantLeases("B")
	require.NoError(t, err)
	require.Equal(t, capPerPool, leasesB)

	stats, err := mp.LeaseStats()
	require.NoError(t, err)
	require.EqualValues(t, 1, stats.Denied)
	require.Equal(t, 2, stats.ActiveTenants)

	close(release)
	require.Eventually(t, func() bool {
		return mp.Running() == 0
	}, 3*time.Second, 5*time.Millisecond)
}

// TestTenantFairWaiterFreesSlot verifies FIFO fairness: a tenant blocked while
// every pool is saturated receives the next slot freed by the incumbents.
func TestTenantFairWaiterFreesSlot(t *testing.T) {
	const poolCount, capPerPool = 2, 2
	mp, err := ants.NewMultiPool(poolCount, capPerPool, ants.TenantFair,
		ants.WithTenantLeaseTTL(time.Minute))
	require.NoError(t, err)
	defer mp.ReleaseTimeout(3 * time.Second)

	// With 2 pools of capacity 2 the per-tenant quota is one pool: tenant A
	// fills pool 0, tenant B fills pool 1, saturating the whole multi-pool.
	runA, releaseA := blockUntil(t)
	for i := 0; i < capPerPool; i++ {
		require.NoError(t, mp.SubmitTenant("A", runA))
	}
	runB, releaseB := blockUntil(t)
	for i := 0; i < capPerPool; i++ {
		require.NoError(t, mp.SubmitTenant("B", runB))
	}

	require.Equal(t, poolCount*capPerPool, mp.Running())
	require.Zero(t, mp.Free())

	tenantCGotSlot := make(chan struct{})
	go func() {
		err := mp.SubmitTenant("C", func() { close(tenantCGotSlot) })
		require.NoError(t, err)
	}()
	require.Eventually(t, func() bool {
		s, _ := mp.LeaseStats()
		return s.Waiting == 1
	}, 3*time.Second, 5*time.Millisecond)

	// Freeing one slot must hand the lease to the waiting tenant C rather
	// than let A or B reclaim it.
	close(releaseA)
	select {
	case <-tenantCGotSlot:
	case <-time.After(3 * time.Second):
		t.Fatal("waiting tenant C did not receive the freed slot")
	}

	close(releaseB)
	require.Eventually(t, func() bool {
		return mp.Running() == 0
	}, 3*time.Second, 5*time.Millisecond)

	stats, err := mp.LeaseStats()
	require.NoError(t, err)
	require.Zero(t, stats.Waiting)
	// Every dispatched task completed and every lease was released.
	require.Equal(t, stats.Dispatched, stats.Completed)
	require.Equal(t, stats.Acquired, stats.Released)
}

// TestTenantFairLeaseTTLExpiry verifies that an expired affinity lease allows
// the tenant to migrate to another pool on its next submission.
func TestTenantFairLeaseTTLExpiry(t *testing.T) {
	const poolCount, capPerPool = 4, 4
	const ttl = 200 * time.Millisecond
	mp, err := ants.NewMultiPool(poolCount, capPerPool, ants.TenantFair,
		ants.WithNonblocking(true), ants.WithTenantLeaseTTL(ttl))
	require.NoError(t, err)
	defer mp.ReleaseTimeout(3 * time.Second)

	// Fill three pools with distinct tenants first; the single remaining pool
	// is where tenant A's sole lease will land.
	run, release := blockUntil(t)
	tenantNo := 0
	for i := 0; i < poolCount-1; i++ {
		tenant := string(rune('B' + tenantNo))
		tenantNo++
		for j := 0; j < capPerPool; j++ {
			require.NoError(t, mp.SubmitTenant(tenant, run))
		}
	}

	// A's single, never-refreshed lease lands on the only free pool.
	require.NoError(t, mp.SubmitTenant("A", run))
	leasesA, err := mp.TenantLeases("A")
	require.NoError(t, err)
	require.Equal(t, 1, leasesA)

	// Wait out the TTL without any further grant for A, so its expireAt is
	// not refreshed; the re-pick must count as expiry-driven and move A.
	time.Sleep(ttl + 100*time.Millisecond)
	require.NoError(t, mp.SubmitTenant("A", run))

	stats, err := mp.LeaseStats()
	require.NoError(t, err)
	require.GreaterOrEqual(t, stats.Expired, int64(1))
	require.Zero(t, stats.Migrated)

	// The expired lease was re-granted within the current generation.
	idx, err := mp.TenantPool("A")
	require.NoError(t, err)
	require.GreaterOrEqual(t, idx, 0)
	leasesA, err = mp.TenantLeases("A")
	require.NoError(t, err)
	require.Equal(t, 2, leasesA)

	close(release)
	require.Eventually(t, func() bool {
		return mp.Running() == 0
	}, 3*time.Second, 5*time.Millisecond)
}

// TestTenantFairSaturationMigration verifies a saturated affinity pool forces
// the tenant onto another pool even while the lease is alive.
func TestTenantFairSaturationMigration(t *testing.T) {
	mp, err := ants.NewMultiPool(3, 2, ants.TenantFair,
		ants.WithNonblocking(true), ants.WithTenantLeaseTTL(time.Minute))
	require.NoError(t, err)
	defer mp.ReleaseTimeout(3 * time.Second)

	run, release := blockUntil(t)
	require.NoError(t, mp.SubmitTenant("A", run))
	firstIdx, err := mp.TenantPool("A")
	require.NoError(t, err)
	require.NoError(t, mp.SubmitTenant("A", run))
	// The affinity pool is now full; the third task must migrate tenant A.
	require.NoError(t, mp.SubmitTenant("A", run))

	stats, err := mp.LeaseStats()
	require.NoError(t, err)
	require.GreaterOrEqual(t, stats.Migrated, int64(1))
	require.Zero(t, stats.Expired)

	idx, err := mp.TenantPool("A")
	require.NoError(t, err)
	require.NotEqual(t, firstIdx, idx)

	close(release)
	require.Eventually(t, func() bool {
		return mp.Running() == 0
	}, 3*time.Second, 5*time.Millisecond)
}

// TestTenantFairGenerationAdvance verifies Tune, ReleaseContext and Reboot
// advance the lease generation and stale leases never touch the new state.
func TestTenantFairGenerationAdvance(t *testing.T) {
	mp, err := ants.NewMultiPool(3, 2, ants.TenantFair, ants.WithNonblocking(true))
	require.NoError(t, err)

	run, release := blockUntil(t)
	require.NoError(t, mp.SubmitTenant("A", run))
	require.NoError(t, mp.SubmitTenant("B", run))

	stats, err := mp.LeaseStats()
	require.NoError(t, err)
	require.Zero(t, stats.Generation)

	// Tune advances the generation while generation-0 tasks are still
	// running; their eventual releases are stale no-ops against new state.
	mp.Tune(4)
	stats, err = mp.LeaseStats()
	require.NoError(t, err)
	require.EqualValues(t, 1, stats.Generation)
	require.Zero(t, stats.ActiveTenants)

	// New leases belong to generation 1 and work normally.
	require.NoError(t, mp.SubmitTenant("A", run))
	idx, err := mp.TenantPool("A")
	require.NoError(t, err)
	require.GreaterOrEqual(t, idx, 0)

	close(release)
	require.Eventually(t, func() bool {
		return mp.Running() == 0
	}, 3*time.Second, 5*time.Millisecond)

	// ReleaseContext + Reboot each advance the generation; the rebooted
	// multi-pool must be usable with clean lease state.
	require.NoError(t, mp.ReleaseTimeout(3*time.Second))
	mp.Reboot()
	stats, err = mp.LeaseStats()
	require.NoError(t, err)
	require.EqualValues(t, 3, stats.Generation)

	run2, release2 := blockUntil(t)
	require.NoError(t, mp.SubmitTenant("C", run2))
	close(release2)
	require.Eventually(t, func() bool {
		return mp.Running() == 0
	}, 3*time.Second, 5*time.Millisecond)
}

// TestTenantFairWithFunc exercises the PoolWithFunc variant, the stats and
// execution accounting, and the unsupported-strategy errors.
func TestTenantFairWithFunc(t *testing.T) {
	var execA, execB int64
	fn := func(arg any) {
		switch tenant := arg.(string); tenant {
		case "A":
			atomic.AddInt64(&execA, 1)
		case "B":
			atomic.AddInt64(&execB, 1)
		}
	}
	mp, err := ants.NewMultiPoolWithFunc(3, 4, fn, ants.TenantFair,
		ants.WithNonblocking(true), ants.WithTenantLeaseTTL(time.Minute))
	require.NoError(t, err)
	defer mp.ReleaseTimeout(3 * time.Second)

	// Tenant APIs on a non-TenantFair multi-pool must fail explicitly.
	rr, err := ants.NewMultiPoolWithFunc(2, 2, fn, ants.RoundRobin)
	require.NoError(t, err)
	defer rr.ReleaseTimeout(3 * time.Second)
	require.ErrorIs(t, rr.InvokeTenant("A", "A"), ants.ErrTenantLeaseUnsupported)
	_, err = rr.LeaseStats()
	require.ErrorIs(t, err, ants.ErrTenantLeaseUnsupported)
	_, err = rr.TenantLeases("A")
	require.ErrorIs(t, err, ants.ErrTenantLeaseUnsupported)
	_, err = rr.TenantPool("A")
	require.ErrorIs(t, err, ants.ErrTenantLeaseUnsupported)

	// Total capacity is 12: interleave two tenants so neither hits its quota
	// and every submission dispatches immediately.
	for i := 0; i < 6; i++ {
		require.NoError(t, mp.InvokeTenant("A", "A"))
		require.NoError(t, mp.InvokeTenant("B", "B"))
	}

	require.Eventually(t, func() bool {
		s, _ := mp.LeaseStats()
		return s.Completed == 12
	}, 3*time.Second, 5*time.Millisecond)
	require.EqualValues(t, 12, atomic.LoadInt64(&execA)+atomic.LoadInt64(&execB))
	require.EqualValues(t, 6, atomic.LoadInt64(&execA))
	require.EqualValues(t, 6, atomic.LoadInt64(&execB))

	stats, err := mp.LeaseStats()
	require.NoError(t, err)
	require.EqualValues(t, 12, stats.Dispatched)
	require.EqualValues(t, 12, stats.Acquired)
	require.EqualValues(t, 12, stats.Released)

	// Plain Invoke routes through the anonymous tenant and still works.
	require.NoError(t, mp.Invoke("A"))
	require.Eventually(t, func() bool {
		s, _ := mp.LeaseStats()
		return s.Completed == 13
	}, 3*time.Second, 5*time.Millisecond)
}

// TestTenantFairGeneric exercises the generic variant end to end.
func TestTenantFairGeneric(t *testing.T) {
	var count int64
	fn := func(d time.Duration) {
		time.Sleep(d)
		atomic.AddInt64(&count, 1)
	}
	mp, err := ants.NewMultiPoolWithFuncGeneric(3, 4, fn, ants.TenantFair,
		ants.WithTenantLeaseTTL(time.Minute))
	require.NoError(t, err)
	defer mp.ReleaseTimeout(3 * time.Second)

	for i := 0; i < 12; i++ {
		require.NoError(t, mp.InvokeTenant("A", time.Millisecond))
	}
	require.Eventually(t, func() bool {
		return atomic.LoadInt64(&count) == 12
	}, 3*time.Second, 5*time.Millisecond)

	stats, err := mp.LeaseStats()
	require.NoError(t, err)
	require.EqualValues(t, 12, stats.Dispatched)
	require.EqualValues(t, 12, stats.Completed)
	require.EqualValues(t, 12, stats.Acquired)
	require.EqualValues(t, 12, stats.Released)
}

// TestTenantFairRejectionLeaksNoLease verifies that a rejected acquisition
// never creates a lease, so no capacity is reserved for the denied tenant.
func TestTenantFairRejectionLeaksNoLease(t *testing.T) {
	mp, err := ants.NewMultiPool(2, 2, ants.TenantFair, ants.WithNonblocking(true))
	require.NoError(t, err)
	defer mp.ReleaseTimeout(3 * time.Second)

	run, release := blockUntil(t)
	// The per-tenant quota of a 2-pool multi-pool is one pool: A and B each
	// fill one pool, saturating every slot.
	for i := 0; i < 2; i++ {
		require.NoError(t, mp.SubmitTenant("A", run))
		require.NoError(t, mp.SubmitTenant("B", run))
	}
	require.Zero(t, mp.Free())

	// The next tenant is denied before any pool is touched, so no lease leaks.
	require.ErrorIs(t, mp.SubmitTenant("C", run), ants.ErrPoolOverload)

	stats, err := mp.LeaseStats()
	require.NoError(t, err)
	require.EqualValues(t, 4, stats.Acquired)
	require.EqualValues(t, 1, stats.Denied)
	leasesC, err := mp.TenantLeases("C")
	require.NoError(t, err)
	require.Zero(t, leasesC)

	close(release)
	require.Eventually(t, func() bool {
		return mp.Running() == 0
	}, 3*time.Second, 5*time.Millisecond)
}

// TestTenantFairRoundRobinUnchanged verifies the default strategies keep their
// original behavior and never instantiate a lease manager.
func TestTenantFairRoundRobinUnchanged(t *testing.T) {
	mp, err := ants.NewMultiPool(4, 2, ants.RoundRobin, ants.WithNonblocking(true))
	require.NoError(t, err)
	defer mp.ReleaseTimeout(3 * time.Second)

	_, err = mp.LeaseStats()
	require.ErrorIs(t, err, ants.ErrTenantLeaseUnsupported)
	require.ErrorIs(t, mp.SubmitTenant("A", func() {}), ants.ErrTenantLeaseUnsupported)

	run, release := blockUntil(t)
	for i := 0; i < 8; i++ {
		require.NoError(t, mp.Submit(run))
	}
	for i := 0; i < 4; i++ {
		n, err := mp.RunningByIndex(i)
		require.NoError(t, err)
		require.Equal(t, 2, n)
	}
	close(release)
	require.Eventually(t, func() bool {
		return mp.Running() == 0
	}, 3*time.Second, 5*time.Millisecond)
}
