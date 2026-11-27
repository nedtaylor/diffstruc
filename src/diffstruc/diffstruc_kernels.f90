module diffstruc__kernels
  !! Elementwise numeric kernels.
  !!
  !! Every kernel takes plain array arguments with explicit `intent`, so the
  !! compiler can prove that the output does not alias the inputs and vectorise
  !! the inner loop. The sample loop carries the optional OpenMP region;
  !! `nth <= 1` makes it an inactive serial region, so the serial build and the
  !! one-thread run execute the same loop.
  !!
  !! No state lives here. The thread count comes from the owning context (see
  !! `kernel_team` in `diffstruc__types`).
  use, intrinsic :: iso_fortran_env, only: real32
  implicit none
  private

  public :: add_two, add_col, sub_two, sub_col
  public :: mul_two, mul_col, mul_row, scale2
  public :: squared2, copy2, axpy2
  public :: mul_right_col, reduce_mul_left, reduce_mul_full, reduce_samples
  public :: sum_columns

contains

  subroutine add_two(out, x, y, n, ns, nth)
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: x, y
    real(real32), dimension(:,:), intent(out) :: out
    integer :: i, s
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       !$omp simd
       do i = 1, n
          out(i,s) = x(i,s) + y(i,s)
       end do
    end do
  end subroutine add_two

  subroutine add_col(out, x, y, n, ns, nth)
    !! out = x + y, where y is a single column (one value per row).
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: x, y
    real(real32), dimension(:,:), intent(out) :: out
    integer :: i, s
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       !$omp simd
       do i = 1, n
          out(i,s) = x(i,s) + y(i,1)
       end do
    end do
  end subroutine add_col

  subroutine sub_two(out, x, y, n, ns, nth)
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: x, y
    real(real32), dimension(:,:), intent(out) :: out
    integer :: i, s
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       !$omp simd
       do i = 1, n
          out(i,s) = x(i,s) - y(i,s)
       end do
    end do
  end subroutine sub_two

  subroutine sub_col(out, x, y, n, ns, nth)
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: x, y
    real(real32), dimension(:,:), intent(out) :: out
    integer :: i, s
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       !$omp simd
       do i = 1, n
          out(i,s) = x(i,s) - y(i,1)
       end do
    end do
  end subroutine sub_col

  subroutine mul_two(out, x, y, n, ns, nth)
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: x, y
    real(real32), dimension(:,:), intent(out) :: out
    integer :: i, s
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       !$omp simd
       do i = 1, n
          out(i,s) = x(i,s) * y(i,s)
       end do
    end do
  end subroutine mul_two

  subroutine mul_col(out, x, y, n, ns, nth)
    !! out = x * y, where y is a single column (one value per row).
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: x, y
    real(real32), dimension(:,:), intent(out) :: out
    integer :: i, s
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       !$omp simd
       do i = 1, n
          out(i,s) = x(i,s) * y(i,1)
       end do
    end do
  end subroutine mul_col

  subroutine mul_row(out, x, vec, n, ns, nth)
    !! out(i,s) = x(i,s) * vec(1,s): one factor per sample.
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: x, vec
    real(real32), dimension(:,:), intent(out) :: out
    integer :: i, s
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       !$omp simd
       do i = 1, n
          out(i,s) = x(i,s) * vec(1,s)
       end do
    end do
  end subroutine mul_row

  subroutine scale2(out, x, factor, n, ns, nth)
    !! out = factor * x for a scalar factor.
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: x
    real(real32), intent(in) :: factor
    real(real32), dimension(:,:), intent(out) :: out
    integer :: i, s
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       !$omp simd
       do i = 1, n
          out(i,s) = factor * x(i,s)
       end do
    end do
  end subroutine scale2

  subroutine squared2(out, x, n, ns, nth)
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: x
    real(real32), dimension(:,:), intent(out) :: out
    integer :: i, s
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       !$omp simd
       do i = 1, n
          out(i,s) = x(i,s) * x(i,s)
       end do
    end do
  end subroutine squared2

  subroutine copy2(out, values, n, ns, nth)
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: values
    real(real32), dimension(:,:), intent(out) :: out
    integer :: i, s
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       !$omp simd
       do i = 1, n
          out(i,s) = values(i,s)
       end do
    end do
  end subroutine copy2

  subroutine axpy2(out, values, n, ns, nth)
    !! out = out + values.
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: values
    real(real32), dimension(:,:), intent(inout) :: out
    integer :: i, s
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       !$omp simd
       do i = 1, n
          out(i,s) = out(i,s) + values(i,s)
       end do
    end do
  end subroutine axpy2

  subroutine mul_right_col(out, upstream, w, n, ns, nth)
    !! out(i,s) = upstream(i,s) * w(i,1).
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: upstream, w
    real(real32), dimension(:,:), intent(out) :: out
    integer :: i, s
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       !$omp simd
       do i = 1, n
          out(i,s) = upstream(i,s) * w(i,1)
       end do
    end do
  end subroutine mul_right_col

  subroutine reduce_mul_left(out, upstream, w, n, ns, nth)
    !! out(i) = sum_s upstream(i,s) * w(i,1), sample order preserved per row.
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: upstream, w
    real(real32), dimension(:), intent(out) :: out
    integer :: i, s, ib, iend
    if(nth > 1) then
       do i = 1, n
          out(i) = upstream(i,1) * w(i,1)
       end do
       !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
       do ib = 1, n, 128
          iend = min(n, ib + 127)
          do s = 2, ns
             !$omp simd
             do i = ib, iend
                out(i) = out(i) + upstream(i,s) * w(i,1)
             end do
          end do
       end do
    else
       out(:) = upstream(:,1) * w(:,1)
       do s = 2, ns
          out(:) = out(:) + upstream(:,s) * w(:,1)
       end do
    end if
  end subroutine reduce_mul_left

  subroutine reduce_mul_full(out, upstream, values, n, ns, nth)
    !! out(i) = sum_s upstream(i,s) * values(i,s), sample order preserved.
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: upstream, values
    real(real32), dimension(:), intent(out) :: out
    integer :: i, s, ib, iend
    if(nth > 1) then
       do i = 1, n
          out(i) = upstream(i,1) * values(i,1)
       end do
       !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
       do ib = 1, n, 128
          iend = min(n, ib + 127)
          do s = 2, ns
             !$omp simd
             do i = ib, iend
                out(i) = out(i) + upstream(i,s) * values(i,s)
             end do
          end do
       end do
    else
       out(:) = upstream(:,1) * values(:,1)
       do s = 2, ns
          out(:) = out(:) + upstream(:,s) * values(:,s)
       end do
    end if
  end subroutine reduce_mul_full

  subroutine reduce_samples(out, upstream, n, ns, nth)
    !! out(i) = sum_s upstream(i,s), sample order preserved.
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: upstream
    real(real32), dimension(:), intent(out) :: out
    integer :: i, s, ib, iend
    if(nth > 1) then
       do i = 1, n
          out(i) = upstream(i,1)
       end do
       !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
       do ib = 1, n, 128
          iend = min(n, ib + 127)
          do s = 2, ns
             !$omp simd
             do i = ib, iend
                out(i) = out(i) + upstream(i,s)
             end do
          end do
       end do
    else
       out(:) = upstream(:,1)
       do s = 2, ns
          out(:) = out(:) + upstream(:,s)
       end do
    end if
  end subroutine reduce_samples

  subroutine sum_columns(out, values, n, ns, nth)
    !! out(s) = sum_i values(i,s). Used by the opt-in parallel reduction.
    integer, intent(in) :: n, ns, nth
    real(real32), dimension(:,:), intent(in) :: values
    real(real32), dimension(:), intent(out) :: out
    integer :: i, s
    real(real32) :: acc
    !$omp parallel do if(nth > 1) num_threads(nth) schedule(static)
    do s = 1, ns
       acc = values(1,s)
       do i = 2, n
          acc = acc + values(i,s)
       end do
       out(s) = acc
    end do
  end subroutine sum_columns

end module diffstruc__kernels
