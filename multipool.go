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
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
)

// LoadBalancingStrategy represents the type of load-balancing algorithm.
type LoadBalancingStrategy int

const (
	// RoundRobin distributes task to a list of pools in rotation.
	RoundRobin LoadBalancingStrategy = 1 << (iota + 1)

	// LeastTasks always selects the pool with the least number of pending tasks.
	LeastTasks

	// TenantFair grants a short-lived capacity lease per tenant on each submit.
	// A tenant keeps affinity to the same pool while its lease is alive and the
	// pool is not saturated, and no tenant can occupy the whole multi-pool:
	// when every pool is saturated, tenants queue fairly and the first queued
	// tenant that can use a freed slot receives the next lease.
	TenantFair
)

// ErrTenantLeaseUnsupported will be returned when tenant-scoped submission is
// called on a multi-pool that was not created with the TenantFair strategy.
var ErrTenantLeaseUnsupported = errors.New("tenant leases require the TenantFair load-balancing strategy")

type contextReleaser interface {
	ReleaseContext(ctx context.Context) error
}

func releasePools(ctx context.Context, pools []contextReleaser) error {
	errCh := make(chan error, len(pools))
	var wg errgroup.Group
	for i, pool := range pools {
		func(p contextReleaser, idx int) {
			wg.Go(func() error {
				err := p.ReleaseContext(ctx)
				if err != nil {
					err = fmt.Errorf("pool %d: %v", idx, err)
				}
				errCh <- err
				return err
			})
		}(pool, i)
	}

	_ = wg.Wait()

	var errStr strings.Builder
	for i := 0; i < len(pools); i++ {
		if err := <-errCh; err != nil {
			errStr.WriteString(err.Error())
			errStr.WriteString(" | ")
		}
	}

	if errStr.Len() == 0 {
		return nil
	}

	return errors.New(strings.TrimSuffix(errStr.String(), " | "))
}

// MultiPool consists of multiple pools, from which you will benefit the
// performance improvement on basis of the fine-grained locking that reduces
// the lock contention.
// MultiPool is a good fit for the scenario where you have a large number of
// tasks to submit, and you don't want the single pool to be the bottleneck.
type MultiPool struct {
	pools []*Pool
	index uint32
	state int32
	lbs   LoadBalancingStrategy

	// leases is only initialized when the multi-pool uses the TenantFair
	// strategy; the RoundRobin and LeastTasks code paths stay lock-free.
	leases      *leaseManager
	nonblocking bool
}

// NewMultiPool instantiates a MultiPool with a size of the pool list and a size
// per pool, and the load-balancing strategy.
func NewMultiPool(size, sizePerPool int, lbs LoadBalancingStrategy, options ...Option) (*MultiPool, error) {
	if size <= 0 {
		return nil, ErrInvalidMultiPoolSize
	}

	if lbs != RoundRobin && lbs != LeastTasks && lbs != TenantFair {
		return nil, ErrInvalidLoadBalancingStrategy
	}
	pools := make([]*Pool, size)
	for i := 0; i < size; i++ {
		pool, err := NewPool(sizePerPool, options...)
		if err != nil {
			// Release all previously created pools to avoid resource leak
			for j := 0; j < i; j++ {
				pools[j].Release()
			}
			return nil, err
		}
		pools[i] = pool
	}
	mp := &MultiPool{pools: pools, index: math.MaxUint32, lbs: lbs}
	if lbs == TenantFair {
		opts := loadOptions(options...)
		mp.nonblocking = opts.Nonblocking
		mp.leases = newLeaseManager(leasePoolSlice(pools), opts.TenantLeaseTTL)
	}
	return mp, nil
}

// leasePoolSlice adapts a concrete pool slice to the leasePool interface.
func leasePoolSlice(pools []*Pool) []leasePool {
	out := make([]leasePool, len(pools))
	for i, p := range pools {
		out[i] = p
	}
	return out
}

func (mp *MultiPool) next(lbs LoadBalancingStrategy) (idx int) {
	switch lbs {
	case RoundRobin:
		return int(atomic.AddUint32(&mp.index, 1) % uint32(len(mp.pools)))
	case LeastTasks:
		leastTasks := math.MaxInt32
		for i, pool := range mp.pools {
			if n := pool.Running(); n < leastTasks {
				leastTasks = n
				idx = i
			}
		}
		return
	}
	return -1
}

// Submit submits a task to a pool selected by the load-balancing strategy.
//
// Under the TenantFair strategy the task is attributed to the anonymous
// tenant; use SubmitTenant to attribute it to a specific tenant.
func (mp *MultiPool) Submit(task func()) (err error) {
	if mp.IsClosed() {
		return ErrPoolClosed
	}
	if mp.lbs == TenantFair {
		return mp.submitLeased(anonymousTenant, task)
	}
	if err = mp.pools[mp.next(mp.lbs)].Submit(task); err == nil {
		return
	}
	if err == ErrPoolOverload && mp.lbs == RoundRobin {
		return mp.pools[mp.next(LeastTasks)].Submit(task)
	}
	return
}

// SubmitTenant submits a task attributed to tenant under the capacity-lease
// fairness rules. The lease is granted before the submission, returned once
// when the submission fails and released when the task finishes executing.
// It returns ErrTenantLeaseUnsupported unless the multi-pool uses TenantFair.
func (mp *MultiPool) SubmitTenant(tenant string, task func()) error {
	if mp.IsClosed() {
		return ErrPoolClosed
	}
	if mp.lbs != TenantFair {
		return ErrTenantLeaseUnsupported
	}
	return mp.submitLeased(tenant, task)
}

// submitLeased drives the acquire -> submit -> release lifecycle of one lease.
func (mp *MultiPool) submitLeased(tenant string, task func()) error {
	l, err := mp.leases.Acquire(tenant, mp.nonblocking)
	if err != nil {
		return err
	}
	// The lease is handed back exactly once: either immediately when the
	// submission fails (allowing a queued tenant to take the slot), or at
	// the end of the task after the worker has picked it up.
	if err = mp.pools[l.poolIdx].Submit(mp.wrapLeased(l, task)); err != nil {
		mp.leases.Release(l)
		return err
	}
	mp.leases.noteDispatched()
	return nil
}

// wrapLeased binds a lease to the task lifecycle: the lease is released when
// the task finishes, even if it panics (the worker's own panic recovery still
// runs afterwards).
func (mp *MultiPool) wrapLeased(l *lease, task func()) func() {
	return func() {
		defer func() {
			mp.leases.noteCompleted()
			mp.leases.Release(l)
		}()
		task()
	}
}

// Running returns the number of the currently running workers across all pools.
func (mp *MultiPool) Running() (n int) {
	for _, pool := range mp.pools {
		n += pool.Running()
	}
	return
}

// RunningByIndex returns the number of the currently running workers in the specific pool.
func (mp *MultiPool) RunningByIndex(idx int) (int, error) {
	if idx < 0 || idx >= len(mp.pools) {
		return -1, ErrInvalidPoolIndex
	}
	return mp.pools[idx].Running(), nil
}

// Free returns the number of available workers across all pools.
func (mp *MultiPool) Free() (n int) {
	for _, pool := range mp.pools {
		n += pool.Free()
	}
	return
}

// FreeByIndex returns the number of available workers in the specific pool.
func (mp *MultiPool) FreeByIndex(idx int) (int, error) {
	if idx < 0 || idx >= len(mp.pools) {
		return -1, ErrInvalidPoolIndex
	}
	return mp.pools[idx].Free(), nil
}

// Waiting returns the number of the currently waiting tasks across all pools.
func (mp *MultiPool) Waiting() (n int) {
	for _, pool := range mp.pools {
		n += pool.Waiting()
	}
	return
}

// WaitingByIndex returns the number of the currently waiting tasks in the specific pool.
func (mp *MultiPool) WaitingByIndex(idx int) (int, error) {
	if idx < 0 || idx >= len(mp.pools) {
		return -1, ErrInvalidPoolIndex
	}
	return mp.pools[idx].Waiting(), nil
}

// Cap returns the capacity of this multi-pool.
func (mp *MultiPool) Cap() (n int) {
	for _, pool := range mp.pools {
		n += pool.Cap()
	}
	return
}

// Tune resizes each pool in multi-pool.
//
// Note that this method doesn't resize the overall
// capacity of multi-pool.
//
// Under the TenantFair strategy tuning advances the lease generation, so
// leases granted before the resize cannot be applied to the resized pools.
func (mp *MultiPool) Tune(size int) {
	for _, pool := range mp.pools {
		pool.Tune(size)
	}
	if mp.leases != nil {
		mp.leases.advanceGeneration()
	}
}

// IsClosed indicates whether the multi-pool is closed.
func (mp *MultiPool) IsClosed() bool {
	return atomic.LoadInt32(&mp.state) == CLOSED
}

// ReleaseTimeout closes the multi-pool with a timeout,
// it waits all pools to be closed before timing out.
func (mp *MultiPool) ReleaseTimeout(timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return mp.ReleaseContext(ctx)
}

// ReleaseContext closes the multi-pool with a context,
// it waits all pools to be closed before the context is done.
func (mp *MultiPool) ReleaseContext(ctx context.Context) error {
	if !atomic.CompareAndSwapInt32(&mp.state, OPENED, CLOSED) {
		return ErrPoolClosed
	}

	if mp.leases != nil {
		// Invalidate in-flight leases and wake up every queued tenant.
		mp.leases.advanceGeneration()
	}

	pools := make([]contextReleaser, len(mp.pools))
	for i, p := range mp.pools {
		pools[i] = p
	}
	return releasePools(ctx, pools)
}

// Reboot reboots a released multi-pool.
func (mp *MultiPool) Reboot() {
	if atomic.CompareAndSwapInt32(&mp.state, CLOSED, OPENED) {
		atomic.StoreUint32(&mp.index, 0)
		if mp.leases != nil {
			// Leases from the previous incarnation must not write into the
			// rebooted pool state.
			mp.leases.advanceGeneration()
		}
		for _, pool := range mp.pools {
			pool.Reboot()
		}
	}
}

// LeaseStats returns the counters of the tenant-fair lease manager. It returns
// ErrTenantLeaseUnsupported unless the multi-pool uses the TenantFair strategy.
func (mp *MultiPool) LeaseStats() (LeaseStats, error) {
	if mp.leases == nil {
		return LeaseStats{}, ErrTenantLeaseUnsupported
	}
	return mp.leases.Stats(), nil
}

// TenantLeases returns the number of outstanding leases (in-flight leased
// tasks) held by tenant.
func (mp *MultiPool) TenantLeases(tenant string) (int, error) {
	if mp.leases == nil {
		return 0, ErrTenantLeaseUnsupported
	}
	return mp.leases.TenantLeases(tenant), nil
}

// TenantPool returns the index of the pool the tenant currently has affinity
// with, or -1 when the tenant holds no lease.
func (mp *MultiPool) TenantPool(tenant string) (int, error) {
	if mp.leases == nil {
		return -1, ErrTenantLeaseUnsupported
	}
	return mp.leases.TenantPool(tenant), nil
}
