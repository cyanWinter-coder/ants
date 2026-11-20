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
	"math"
	"sync/atomic"
	"time"
)

// MultiPoolWithFunc consists of multiple pools, from which you will benefit the
// performance improvement on basis of the fine-grained locking that reduces
// the lock contention.
// MultiPoolWithFunc is a good fit for the scenario where you have a large number of
// tasks to submit, and you don't want the single pool to be the bottleneck.
type MultiPoolWithFunc struct {
	pools []*PoolWithFunc
	index uint32
	state int32
	lbs   LoadBalancingStrategy

	// leases is only initialized when the multi-pool uses the TenantFair
	// strategy; the RoundRobin and LeastTasks code paths stay lock-free.
	leases      *leaseManager
	nonblocking bool
}

// leasedFuncJob carries a tenant task through PoolWithFunc.Invoke. It is an
// unexported type so it can never collide with user-provided arguments.
type leasedFuncJob struct {
	lease *lease
	arg   any
	mp    *MultiPoolWithFunc
}

// NewMultiPoolWithFunc instantiates a MultiPoolWithFunc with a size of the pool list and a size
// per pool, and the load-balancing strategy.
func NewMultiPoolWithFunc(size, sizePerPool int, fn func(any), lbs LoadBalancingStrategy, options ...Option) (*MultiPoolWithFunc, error) {
	if size <= 0 {
		return nil, ErrInvalidMultiPoolSize
	}

	if lbs != RoundRobin && lbs != LeastTasks && lbs != TenantFair {
		return nil, ErrInvalidLoadBalancingStrategy
	}

	wrappedFn := fn
	if lbs == TenantFair {
		// Tenant submissions arrive as *leasedFuncJob; unwrap them, run the
		// user function and release the lease at task completion.
		wrappedFn = func(arg any) {
			if job, ok := arg.(*leasedFuncJob); ok {
				defer func() {
					job.mp.leases.noteCompleted()
					job.mp.leases.Release(job.lease)
				}()
				fn(job.arg)
				return
			}
			fn(arg)
		}
	}

	pools := make([]*PoolWithFunc, size)
	for i := 0; i < size; i++ {
		pool, err := NewPoolWithFunc(sizePerPool, wrappedFn, options...)
		if err != nil {
			// Release all previously created pools to avoid resource leak
			for j := 0; j < i; j++ {
				pools[j].Release()
			}
			return nil, err
		}
		pools[i] = pool
	}
	mp := &MultiPoolWithFunc{pools: pools, index: math.MaxUint32, lbs: lbs}
	if lbs == TenantFair {
		opts := loadOptions(options...)
		mp.nonblocking = opts.Nonblocking
		mp.leases = newLeaseManager(leaseFuncPoolSlice(pools), opts.TenantLeaseTTL)
	}
	return mp, nil
}

// leaseFuncPoolSlice adapts a concrete pool slice to the leasePool interface.
func leaseFuncPoolSlice(pools []*PoolWithFunc) []leasePool {
	out := make([]leasePool, len(pools))
	for i, p := range pools {
		out[i] = p
	}
	return out
}

func (mp *MultiPoolWithFunc) next(lbs LoadBalancingStrategy) (idx int) {
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

// Invoke submits a task to a pool selected by the load-balancing strategy.
//
// Under the TenantFair strategy the task is attributed to the anonymous
// tenant; use InvokeTenant to attribute it to a specific tenant.
func (mp *MultiPoolWithFunc) Invoke(args any) (err error) {
	if mp.IsClosed() {
		return ErrPoolClosed
	}
	if mp.lbs == TenantFair {
		return mp.invokeLeased(anonymousTenant, args)
	}

	if err = mp.pools[mp.next(mp.lbs)].Invoke(args); err == nil {
		return
	}
	if err == ErrPoolOverload && mp.lbs == RoundRobin {
		return mp.pools[mp.next(LeastTasks)].Invoke(args)
	}
	return
}

// InvokeTenant submits a task attributed to tenant under the capacity-lease
// fairness rules. The lease is granted before the submission, returned once
// when the submission fails and released when the task finishes executing.
// It returns ErrTenantLeaseUnsupported unless the multi-pool uses TenantFair.
func (mp *MultiPoolWithFunc) InvokeTenant(tenant string, args any) error {
	if mp.IsClosed() {
		return ErrPoolClosed
	}
	if mp.lbs != TenantFair {
		return ErrTenantLeaseUnsupported
	}
	return mp.invokeLeased(tenant, args)
}

// invokeLeased drives the acquire -> invoke -> release lifecycle of one lease.
func (mp *MultiPoolWithFunc) invokeLeased(tenant string, args any) error {
	l, err := mp.leases.Acquire(tenant, mp.nonblocking)
	if err != nil {
		return err
	}
	// The lease is handed back exactly once: either immediately when the
	// invocation fails (allowing a queued tenant to take the slot), or at
	// the end of the task inside the wrapped pool function.
	if err = mp.pools[l.poolIdx].Invoke(&leasedFuncJob{lease: l, arg: args, mp: mp}); err != nil {
		mp.leases.Release(l)
		return err
	}
	mp.leases.noteDispatched()
	return nil
}

// Running returns the number of the currently running workers across all pools.
func (mp *MultiPoolWithFunc) Running() (n int) {
	for _, pool := range mp.pools {
		n += pool.Running()
	}
	return
}

// RunningByIndex returns the number of the currently running workers in the specific pool.
func (mp *MultiPoolWithFunc) RunningByIndex(idx int) (int, error) {
	if idx < 0 || idx >= len(mp.pools) {
		return -1, ErrInvalidPoolIndex
	}
	return mp.pools[idx].Running(), nil
}

// Free returns the number of available workers across all pools.
func (mp *MultiPoolWithFunc) Free() (n int) {
	for _, pool := range mp.pools {
		n += pool.Free()
	}
	return
}

// FreeByIndex returns the number of available workers in the specific pool.
func (mp *MultiPoolWithFunc) FreeByIndex(idx int) (int, error) {
	if idx < 0 || idx >= len(mp.pools) {
		return -1, ErrInvalidPoolIndex
	}
	return mp.pools[idx].Free(), nil
}

// Waiting returns the number of the currently waiting tasks across all pools.
func (mp *MultiPoolWithFunc) Waiting() (n int) {
	for _, pool := range mp.pools {
		n += pool.Waiting()
	}
	return
}

// WaitingByIndex returns the number of the currently waiting tasks in the specific pool.
func (mp *MultiPoolWithFunc) WaitingByIndex(idx int) (int, error) {
	if idx < 0 || idx >= len(mp.pools) {
		return -1, ErrInvalidPoolIndex
	}
	return mp.pools[idx].Waiting(), nil
}

// Cap returns the capacity of this multi-pool.
func (mp *MultiPoolWithFunc) Cap() (n int) {
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
func (mp *MultiPoolWithFunc) Tune(size int) {
	for _, pool := range mp.pools {
		pool.Tune(size)
	}
	if mp.leases != nil {
		mp.leases.advanceGeneration()
	}
}

// IsClosed indicates whether the multi-pool is closed.
func (mp *MultiPoolWithFunc) IsClosed() bool {
	return atomic.LoadInt32(&mp.state) == CLOSED
}

// ReleaseTimeout closes the multi-pool with a timeout,
// it waits all pools to be closed before timing out.
func (mp *MultiPoolWithFunc) ReleaseTimeout(timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return mp.ReleaseContext(ctx)
}

// ReleaseContext closes the multi-pool with a context,
// it waits all pools to be closed before the context is done.
func (mp *MultiPoolWithFunc) ReleaseContext(ctx context.Context) error {
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
func (mp *MultiPoolWithFunc) Reboot() {
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
func (mp *MultiPoolWithFunc) LeaseStats() (LeaseStats, error) {
	if mp.leases == nil {
		return LeaseStats{}, ErrTenantLeaseUnsupported
	}
	return mp.leases.Stats(), nil
}

// TenantLeases returns the number of outstanding leases (in-flight leased
// tasks) held by tenant.
func (mp *MultiPoolWithFunc) TenantLeases(tenant string) (int, error) {
	if mp.leases == nil {
		return 0, ErrTenantLeaseUnsupported
	}
	return mp.leases.TenantLeases(tenant), nil
}

// TenantPool returns the index of the pool the tenant currently has affinity
// with, or -1 when the tenant holds no lease.
func (mp *MultiPoolWithFunc) TenantPool(tenant string) (int, error) {
	if mp.leases == nil {
		return -1, ErrTenantLeaseUnsupported
	}
	return mp.leases.TenantPool(tenant), nil
}
