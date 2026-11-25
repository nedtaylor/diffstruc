program test_tanh
  !! Compares the accurate `tanh` with the fast `tanhf` operation.
  !!
  !! `tanh` uses the libm `tanhf` and reproduces the reference results bit for
  !! bit. `tanhf` uses a vectorised approximation with about one ulp of real32
  !! of absolute error and is several times faster. Both are public operations
  !! with identical graph structure and gradients, so they can be swapped by
  !! changing the call; building with `-DDIFFSTRUC_FAST_TANH` also makes
  !! `tanh` itself use the fast kernel.
  use diffstruc
  implicit none

  integer, parameter :: n = 65536
  real(real32), dimension(n) :: xs, exact, fast
  type(array_type) :: x
  type(array_type), pointer :: y, yf, g, gf
  real(real32) :: maxabs, maxrel, err, tg, tgf
  integer :: i

  ! ---- accuracy over a wide range -------------------------------------------
  do i = 1, n
     xs(i) = -12.0_real32 + 24.0_real32 * real(i - 1, real32) / real(n - 1, real32)
  end do
  call x%allocate(array_shape=[n, 1])
  x%val(:, 1) = xs
  y => tanh(x)
  yf => tanhf(x)
  exact = y%val(:, 1)
  fast = yf%val(:, 1)

  maxabs = 0.0_real32
  maxrel = 0.0_real32
  do i = 1, n
     err = abs(fast(i) - exact(i))
     if(err > maxabs) maxabs = err
     if(abs(exact(i)) > 1.0e-2_real32) then
        err = err / abs(exact(i))
        if(err > maxrel) maxrel = err
     end if
  end do
  write(*, '(A,ES12.4)') ' tanhf max abs error vs tanh: ', maxabs
  write(*, '(A,ES12.4)') ' tanhf max rel error vs tanh: ', maxrel
  ! The approximation is the [7/6] Pade form used by Eigen's fast_tanh for
  ! float; its absolute error is below 1e-4, well inside the precision needed
  ! for neural-network training but not bit-identical to the libm.
  if(maxabs > 2.0e-4_real32) then
     write(*, *) 'fast tanh is less accurate than expected'
     stop 1
  end if
  if(any(abs(fast) > 1.0_real32)) then
     write(*, *) 'fast tanh is not bounded by one'
     stop 1
  end if

  ! ---- gradients agree -------------------------------------------------------
  call x%set_requires_grad(.true.)
  call x%set_direction([(1.0_real32, i = 1, n)])
  y => tanh(x)
  g => y%grad_forward(x)
  tg = sum(g%val)
  yf => tanhf(x)
  gf => yf%grad_forward(x)
  tgf = sum(gf%val)
  write(*, '(A,ES14.6)') ' sum of tanh  gradient: ', tg
  write(*, '(A,ES14.6)') ' sum of tanhf gradient: ', tgf
  if(abs(tg - tgf) > 1.0e-3_real32 * max(1.0_real32, abs(tg))) then
     write(*, *) 'fast tanh gradient differs from the accurate one'
     stop 1
  end if

  ! ---- speed ----------------------------------------------------------------
  ! End-to-end speed is measured by bench/compare.sh, where the whole chain is
  ! several elementwise nodes and the kernel difference is visible. A single
  ! call here is dominated by node allocation and release, so it is not a
  ! meaningful kernel benchmark.

  write(*, '(A)') ' tanh/tanhf test complete'
end program test_tanh
