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

package ants

import (
	"sync"
	"time"
)

// defaultTenantLeaseTTL is the default lifetime of a tenant's capacity lease.
// Within the TTL a tenant keeps its affinity to the same pool; once the lease
// expires the tenant may be placed onto another pool.
const defaultTenantLeaseTTL = time.Second

// anonymousTenant is the tenant id used when plain Submit/Invoke is called on
// a multi-pool configured with the TenantFair strategy.
const anonymousTenant = ""

// leasePool is the subset of the pool surface that the lease manager relies on.
type leasePool interface {
	// Running returns the number of workers currently running in the pool.
	Running() int
	// Cap returns the capacity of the pool, a negative value means unlimited.
	Cap() int
}

// lease is a short-lived capacity reservation granted to a tenant for a single
// pool. A lease belongs to a generation: once the generation is advanced (Tune,
// ReleaseContext, Reboot) the lease becomes stale and can no longer be applied
// to the pool state.
type lease struct {
	tenant     string
	poolIdx    int
	generation uint64
}

// tenantLease records the affinity of a tenant within the current generation.
type tenantLease struct {
	poolIdx  int
	acquired int64 // number of outstanding leases held by the tenant
	expireAt int64 // unix nano after which the affinity may be migrated
}

// leaseWaiter represents a tenant submission queued until a free slot shows up.
// A granted lease is delivered through ch; ch being closed means the generation
// was advanced while waiting and the caller must retry.
type leaseWaiter struct {
	tenant string
	ch     chan *lease
}

// LeaseStats exposes counters that make the tenant-fair strategy auditable.
// All values are snapshots taken at the time LeaseStats is called.
type LeaseStats struct {
	// Generation is the current lease generation, it is advanced by Tune,
	// ReleaseContext and Reboot.
	Generation uint64
	// ActiveTenants is the number of tenants currently holding leases.
	ActiveTenants int
	// Acquired counts every lease granted to a tenant.
	Acquired int64
	// Renewed counts grants that re-used the tenant's affinity pool.
	Renewed int64
	// Migrated counts grants that moved a tenant to another pool because the
	// affinity pool was saturated or the tenant reached its fairness quota.
	Migrated int64
	// Expired counts grants that re-selected a pool because the affinity lease
	// had exceeded the TTL.
	Expired int64
	// Released counts leases returned after a task finished or a submission failed.
	Released int64
	// Denied counts acquisition attempts rejected because no slot was available
	// in nonblocking mode.
	Denied int64
	// Waiting is the number of tenants currently queued for a free slot.
	Waiting int
	// Dispatched counts tasks successfully submitted to a worker under a lease.
	Dispatched int64
	// Completed counts leased tasks that have finished executing.
	Completed int64
}

// leaseManager implements tenant-fair capacity leasing on top of a fixed list
// of pools. It is safe for concurrent use.
//
// Fairness rules:
//   - a tenant keeps affinity to one pool while its lease is alive and the pool
//     is not saturated;
//   - a tenant can never occupy the whole capacity of the multi-pool, at least
//     one pool's worth of capacity is always left for the other tenants;
//   - when every pool is saturated, tenants queue in FIFO order and the first
//     waiter that can make use of a freed slot receives the new lease, so a
//     long-waiting tenant gets the next opening rather than the incumbent.
type leaseManager struct {
	mu sync.Mutex

	pools    []leasePool
	ttl      int64 // unix nano
	gen      uint64
	rrCursor int // rotating starting point when scanning for an eligible pool

	tenants map[string]*tenantLease
	// poolLeases[i] is the number of outstanding leases against pools[i].
	poolLeases []int
	waiters    []*leaseWaiter

	stats LeaseStats
}

func newLeaseManager(pools []leasePool, ttl time.Duration) *leaseManager {
	if ttl <= 0 {
		ttl = defaultTenantLeaseTTL
	}
	return &leaseManager{
		pools:      pools,
		ttl:        int64(ttl),
		tenants:    make(map[string]*tenantLease),
		poolLeases: make([]int, len(pools)),
	}
}

// tenantQuota returns the maximum number of leases a single tenant may hold at
// once. It always leaves at least one whole pool's worth of capacity to the
// other tenants.
func (lm *leaseManager) tenantQuota() int {
	poolCount := len(lm.pools)
	capPerPool := lm.pools[0].Cap()
	if capPerPool < 0 {
		// Unlimited pools cannot be saturated, bound the number of outstanding
		// leases a single tenant may accumulate instead.
		return poolCount
	}
	if poolCount == 1 {
		return capPerPool
	}
	return (poolCount - 1) * capPerPool
}

// poolSaturated reports whether pool idx has no free slot left. Unlimited
// pools are never saturated.
func (lm *leaseManager) poolSaturated(idx int) bool {
	capacity := lm.pools[idx].Cap()
	if capacity < 0 {
		return false
	}
	return lm.poolLeases[idx] >= capacity
}

// pickPool returns the least leased eligible pool, rotating the scan start so
// that equally eligible pools are picked in turn. It returns -1 when every
// pool is saturated or the tenant has reached its fairness quota.
func (lm *leaseManager) pickPool(tenantHeld int64) int {
	if int(tenantHeld) >= lm.tenantQuota() {
		return -1
	}
	poolCount := len(lm.pools)
	least := -1
	leastLeases := 0
	for step := 0; step < poolCount; step++ {
		idx := (lm.rrCursor + step) % poolCount
		if lm.poolSaturated(idx) {
			continue
		}
		if least == -1 || lm.poolLeases[idx] < leastLeases {
			least = idx
			leastLeases = lm.poolLeases[idx]
		}
	}
	if least != -1 {
		lm.rrCursor = (least + 1) % poolCount
	}
	return least
}

// grantLocked hands a fresh lease of the current generation to pool idx for
// tenant and updates every related counter. Caller must hold lm.mu.
func (lm *leaseManager) grantLocked(tenant string, idx int) *lease {
	now := time.Now().UnixNano()
	tl, ok := lm.tenants[tenant]
	if !ok {
		tl = &tenantLease{poolIdx: idx}
		lm.tenants[tenant] = tl
	}
	tl.poolIdx = idx
	tl.acquired++
	tl.expireAt = now + lm.ttl
	lm.poolLeases[idx]++
	lm.stats.Acquired++
	return &lease{tenant: tenant, poolIdx: idx, generation: lm.gen}
}

// tryGrantHeadLocked hands a freed slot on pool idx to the first queued tenant
// that is allowed to use it. The granted lease is returned together with the
// waiter that must be notified; nil means no waiter could take the slot.
func (lm *leaseManager) tryGrantHeadLocked(idx int) (*leaseWaiter, *lease) {
	for head := 0; head < len(lm.waiters); head++ {
		w := lm.waiters[head]
		var held int64
		if tl, ok := lm.tenants[w.tenant]; ok {
			held = tl.acquired
		}
		if int(held) >= lm.tenantQuota() || lm.poolSaturated(idx) {
			continue
		}
		lm.waiters = append(lm.waiters[:head], lm.waiters[head+1:]...)
		return w, lm.grantLocked(w.tenant, idx)
	}
	return nil, nil
}

// Acquire reserves a capacity slot for tenant. In blocking mode it queues FIFO
// until a slot is handed over by a Release or the generation advances; in
// nonblocking mode it returns ErrPoolOverload when nothing is available.
func (lm *leaseManager) Acquire(tenant string, nonblocking bool) (*lease, error) {
	for {
		lm.mu.Lock()

		tl, hasAffinity := lm.tenants[tenant]
		now := time.Now().UnixNano()
		if hasAffinity && tl.acquired > 0 && now < tl.expireAt &&
			!lm.poolSaturated(tl.poolIdx) && int(tl.acquired) < lm.tenantQuota() {
			// Fast path: the affinity lease is alive, its pool still has room
			// and the tenant is below its fairness quota.
			l := lm.grantLocked(tenant, tl.poolIdx)
			lm.stats.Renewed++
			lm.mu.Unlock()
			return l, nil
		}

		var held int64
		// Capture the expiry verdict before granting: grantLocked refreshes
		// expireAt on the same tenantLease, so reading it afterwards would
		// always look "not expired".
		expired := hasAffinity && tl.acquired > 0 && now >= tl.expireAt
		if hasAffinity {
			held = tl.acquired
		}
		idx := lm.pickPool(held)
		if idx != -1 {
			l := lm.grantLocked(tenant, idx)
			if hasAffinity && held > 0 {
				// Expiry takes precedence: even when the affinity pool is
				// saturated at the same time, the re-pick is driven by the
				// expired lease rather than counted as a plain migration.
				if expired {
					lm.stats.Expired++
				} else {
					lm.stats.Migrated++
				}
			}
			lm.mu.Unlock()
			return l, nil
		}

		if nonblocking {
			lm.stats.Denied++
			lm.mu.Unlock()
			return nil, ErrPoolOverload
		}

		w := &leaseWaiter{tenant: tenant, ch: make(chan *lease, 1)}
		lm.waiters = append(lm.waiters, w)
		lm.mu.Unlock()

		// Block until the slot is granted directly or the generation advances,
		// in which case ch is closed and we retry against the new generation.
		l, open := <-w.ch
		if !open {
			continue
		}
		return l, nil
	}
}

// Release returns a lease. It is a no-op for stale leases, so callers can
// always release safely after Tune/ReleaseContext/Reboot. When tenants are
// queued, the freed slot is handed over to the first waiter that can use it.
func (lm *leaseManager) Release(l *lease) {
	if l == nil {
		return
	}
	lm.mu.Lock()
	defer lm.mu.Unlock()

	if l.generation != lm.gen {
		// Stale lease: its counters were already dropped when the generation
		// advanced, it must not touch the new pool state.
		return
	}

	tl, ok := lm.tenants[l.tenant]
	if !ok {
		return
	}
	tl.acquired--
	if tl.acquired <= 0 {
		delete(lm.tenants, l.tenant)
	}
	lm.poolLeases[l.poolIdx]--
	lm.stats.Released++

	if w, granted := lm.tryGrantHeadLocked(l.poolIdx); w != nil {
		// ch has a buffer of one and the receiver is already parked on it,
		// so this send never blocks while lm.mu is held.
		w.ch <- granted
	}
}

// advanceGeneration invalidates every outstanding lease and wakes all queued
// waiters. Old leases become stale and their Release is a no-op, so they can
// never write into the new pool state.
func (lm *leaseManager) advanceGeneration() {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	lm.gen++
	lm.tenants = make(map[string]*tenantLease)
	lm.poolLeases = make([]int, len(lm.pools))
	waiters := lm.waiters
	lm.waiters = nil
	for _, w := range waiters {
		close(w.ch)
	}
}

// noteDispatched records one task successfully handed to a worker.
func (lm *leaseManager) noteDispatched() {
	lm.mu.Lock()
	lm.stats.Dispatched++
	lm.mu.Unlock()
}

// noteCompleted records one leased task finishing its execution.
func (lm *leaseManager) noteCompleted() {
	lm.mu.Lock()
	lm.stats.Completed++
	lm.mu.Unlock()
}

// Stats returns a point-in-time snapshot of the lease counters.
func (lm *leaseManager) Stats() LeaseStats {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	s := lm.stats
	s.Generation = lm.gen
	s.ActiveTenants = len(lm.tenants)
	s.Waiting = len(lm.waiters)
	return s
}

// TenantLeases returns the number of outstanding leases held by tenant.
func (lm *leaseManager) TenantLeases(tenant string) int {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	if tl, ok := lm.tenants[tenant]; ok {
		return int(tl.acquired)
	}
	return 0
}

// TenantPool returns the pool index the tenant currently has affinity with,
// or -1 when the tenant holds no lease.
func (lm *leaseManager) TenantPool(tenant string) int {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	if tl, ok := lm.tenants[tenant]; ok && tl.acquired > 0 {
		return tl.poolIdx
	}
	return -1
}
