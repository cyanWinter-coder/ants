// MIT License

// Copyright (c) 2025 Andy Pan

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

// managedGenericPool is the common management surface of the concrete generic
// pool types, regardless of their argument type.
type managedGenericPool interface {
	leasePool
	Free() int
	Waiting() int
	Tune(int)
	Release()
	Reboot()
	ReleaseContext(context.Context) error
}

// MultiPoolWithFuncGeneric is the generic version of MultiPoolWithFunc.
type MultiPoolWithFuncGeneric[T any] struct {
	// pools is used by the RoundRobin and LeastTasks strategies.
	pools []*PoolWithFuncGeneric[T]
	// leasedPools is used by the TenantFair strategy; their arguments are
	// leased jobs wrapping the user argument, so the lease can be released
	// at task completion without changing the public fn signature.
	leasedPools []*PoolWithFuncGeneric[*leasedGenericJob[T]]
	// managed covers every pool for the strategy-agnostic read and lifecycle
	// methods.
	managed []managedGenericPool

	index uint32
	state int32
	lbs   LoadBalancingStrategy

	// leases is only initialized when the multi-pool uses the TenantFair
	// strategy; the RoundRobin and LeastTasks code paths stay lock-free.
	leases      *leaseManager
	nonblocking bool
}

// leasedGenericJob carries a tenant task through the internal generic pool.
type leasedGenericJob[T any] struct {
	lease *lease
	arg   T
}

// NewMultiPoolWithFuncGeneric instantiates a MultiPoolWithFunc with a size of the pool list and a size
// per pool, and the load-balancing strategy.
func NewMultiPoolWithFuncGeneric[T any](size, sizePerPool int, fn func(T), lbs LoadBalancingStrategy, options ...Option) (*MultiPoolWithFuncGeneric[T], error) {
	if size <= 0 {
		return nil, ErrInvalidMultiPoolSize
	}

	if lbs != RoundRobin && lbs != LeastTasks && lbs != TenantFair {
		return nil, ErrInvalidLoadBalancingStrategy
	}

	mp := &MultiPoolWithFuncGeneric[T]{index: math.MaxUint32, lbs: lbs}

	if lbs == TenantFair {
		// Internal pools process leased jobs: run the user function on the
		// wrapped argument and release the lease at task completion.
		wrappedFn := func(job *leasedGenericJob[T]) {
			defer func() {
				mp.leases.noteCompleted()
				mp.leases.Release(job.lease)
			}()
			fn(job.arg)
		}
		mp.leasedPools = make([]*PoolWithFuncGeneric[*leasedGenericJob[T]], size)
		mp.managed = make([]managedGenericPool, size)
		for i := 0; i < size; i++ {
			pool, err := NewPoolWithFuncGeneric(sizePerPool, wrappedFn, options...)
			if err != nil {
				for j := 0; j < i; j++ {
					mp.leasedPools[j].Release()
				}
				return nil, err
			}
			mp.leasedPools[i] = pool
			mp.managed[i] = pool
		}
		opts := loadOptions(options...)
		mp.nonblocking = opts.Nonblocking
		mp.leases = newLeaseManager(leaseManagedSlice(mp.managed), opts.TenantLeaseTTL)
		return mp, nil
	}

	mp.pools = make([]*PoolWithFuncGeneric[T], size)
	mp.managed = make([]managedGenericPool, size)
	for i := 0; i < size; i++ {
		pool, err := NewPoolWithFuncGeneric(sizePerPool, fn, options...)
		if err != nil {
			// Release all previously created pools to avoid resource leak
			for j := 0; j < i; j++ {
				mp.pools[j].Release()
			}
			return nil, err
		}
		mp.pools[i] = pool
		mp.managed[i] = pool
	}
	return mp, nil
}

// leaseManagedSlice adapts a managed pool slice to the leasePool interface.
func leaseManagedSlice(pools []managedGenericPool) []leasePool {
	out := make([]leasePool, len(pools))
	for i, p := range pools {
		out[i] = p
	}
	return out
}

func (mp *MultiPoolWithFuncGeneric[T]) next(lbs LoadBalancingStrategy) (idx int) {
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
func (mp *MultiPoolWithFuncGeneric[T]) Invoke(args T) (err error) {
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
func (mp *MultiPoolWithFuncGeneric[T]) InvokeTenant(tenant string, args T) error {
	if mp.IsClosed() {
		return ErrPoolClosed
	}
	if mp.lbs != TenantFair {
		return ErrTenantLeaseUnsupported
	}
	return mp.invokeLeased(tenant, args)
}

// invokeLeased drives the acquire -> invoke -> release lifecycle of one lease.
func (mp *MultiPoolWithFuncGeneric[T]) invokeLeased(tenant string, args T) error {
	l, err := mp.leases.Acquire(tenant, mp.nonblocking)
	if err != nil {
		return err
	}
	job := &leasedGenericJob[T]{lease: l, arg: args}
	// The lease is handed back exactly once: either immediately when the
	// invocation fails (allowing a queued tenant to take the slot), or at
	// the end of the task inside the wrapped pool function.
	if err = mp.leasedPools[l.poolIdx].Invoke(job); err != nil {
		mp.leases.Release(l)
		return err
	}
	mp.leases.noteDispatched()
	return nil
}

// Running returns the number of the currently running workers across all pools.
func (mp *MultiPoolWithFuncGeneric[T]) Running() (n int) {
	for _, pool := range mp.managed {
		n += pool.Running()
	}
	return
}

// RunningByIndex returns the number of the currently running workers in the specific pool.
func (mp *MultiPoolWithFuncGeneric[T]) RunningByIndex(idx int) (int, error) {
	if idx < 0 || idx >= len(mp.managed) {
		return -1, ErrInvalidPoolIndex
	}
	return mp.managed[idx].Running(), nil
}

// Free returns the number of available workers across all pools.
func (mp *MultiPoolWithFuncGeneric[T]) Free() (n int) {
	for _, pool := range mp.managed {
		n += pool.Free()
	}
	return
}

// FreeByIndex returns the number of available workers in the specific pool.
func (mp *MultiPoolWithFuncGeneric[T]) FreeByIndex(idx int) (int, error) {
	if idx < 0 || idx >= len(mp.managed) {
		return -1, ErrInvalidPoolIndex
	}
	return mp.managed[idx].Free(), nil
}

// Waiting returns the number of the currently waiting tasks across all pools.
func (mp *MultiPoolWithFuncGeneric[T]) Waiting() (n int) {
	for _, pool := range mp.managed {
		n += pool.Waiting()
	}
	return
}

// WaitingByIndex returns the number of the currently waiting tasks in the specific pool.
func (mp *MultiPoolWithFuncGeneric[T]) WaitingByIndex(idx int) (int, error) {
	if idx < 0 || idx >= len(mp.managed) {
		return -1, ErrInvalidPoolIndex
	}
	return mp.managed[idx].Waiting(), nil
}

// Cap returns the capacity of this multi-pool.
func (mp *MultiPoolWithFuncGeneric[T]) Cap() (n int) {
	for _, pool := range mp.managed {
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
func (mp *MultiPoolWithFuncGeneric[T]) Tune(size int) {
	for _, pool := range mp.managed {
		pool.Tune(size)
	}
	if mp.leases != nil {
		mp.leases.advanceGeneration()
	}
}

// IsClosed indicates whether the multi-pool is closed.
func (mp *MultiPoolWithFuncGeneric[T]) IsClosed() bool {
	return atomic.LoadInt32(&mp.state) == CLOSED
}

// ReleaseTimeout closes the multi-pool with a timeout,
// it waits all pools to be closed before timing out.
func (mp *MultiPoolWithFuncGeneric[T]) ReleaseTimeout(timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return mp.ReleaseContext(ctx)
}

// ReleaseContext closes the multi-pool with a context,
// it waits all pools to be closed before the context is done.
func (mp *MultiPoolWithFuncGeneric[T]) ReleaseContext(ctx context.Context) error {
	if !atomic.CompareAndSwapInt32(&mp.state, OPENED, CLOSED) {
		return ErrPoolClosed
	}

	if mp.leases != nil {
		// Invalidate in-flight leases and wake up every queued tenant.
		mp.leases.advanceGeneration()
	}

	pools := make([]contextReleaser, len(mp.managed))
	for i, p := range mp.managed {
		pools[i] = p
	}
	return releasePools(ctx, pools)
}

// Reboot reboots a released multi-pool.
func (mp *MultiPoolWithFuncGeneric[T]) Reboot() {
	if atomic.CompareAndSwapInt32(&mp.state, CLOSED, OPENED) {
		atomic.StoreUint32(&mp.index, 0)
		if mp.leases != nil {
			// Leases from the previous incarnation must not write into the
			// rebooted pool state.
			mp.leases.advanceGeneration()
		}
		for _, pool := range mp.managed {
			pool.Reboot()
		}
	}
}

// LeaseStats returns the counters of the tenant-fair lease manager. It returns
// ErrTenantLeaseUnsupported unless the multi-pool uses the TenantFair strategy.
func (mp *MultiPoolWithFuncGeneric[T]) LeaseStats() (LeaseStats, error) {
	if mp.leases == nil {
		return LeaseStats{}, ErrTenantLeaseUnsupported
	}
	return mp.leases.Stats(), nil
}

// TenantLeases returns the number of outstanding leases (in-flight leased
// tasks) held by tenant.
func (mp *MultiPoolWithFuncGeneric[T]) TenantLeases(tenant string) (int, error) {
	if mp.leases == nil {
		return 0, ErrTenantLeaseUnsupported
	}
	return mp.leases.TenantLeases(tenant), nil
}

// TenantPool returns the index of the pool the tenant currently has affinity
// with, or -1 when the tenant holds no lease.
func (mp *MultiPoolWithFuncGeneric[T]) TenantPool(tenant string) (int, error) {
	if mp.leases == nil {
		return -1, ErrTenantLeaseUnsupported
	}
	return mp.leases.TenantPool(tenant), nil
}
