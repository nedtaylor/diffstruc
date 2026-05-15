submodule(diffstruc__operations_linalg) diffstruc__operations_linalg_sub
  !! Submodule containing implementations of linear algebra operations
  use diffstruc__backend_linalg, only: backend_sgemm, backend_sgemv
  use coreutils, only: stop_program

contains

!###############################################################################
  module function matmul_arrays(a, b) result(c)
    !! Matrix multiplication of two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s, m, k, n, num_samples
    character(len=128) :: err_msg
    real(real32), pointer :: temp(:,:)

    if(.not.a%is_sample_dependent)then
       if(size(b%shape).ne.1)then
          write(err_msg,'("Matrix multiplication not implemented for array ''b'' &
               &rank: ",I0)') size(b%shape)
          call stop_program(err_msg)
          return
       end if
       ! C(m, S) = A(m, k) * B(k, S) where A is the weight matrix
       m = a%shape(1)
       k = a%shape(2)
       num_samples = size(b%val, 2)
       c => a%create_result(array_shape=[m, num_samples])
       temp(1:m, 1:k) => a%val
#ifdef USE_BLAS
       ! sgemm: C = alpha * A * B + beta * C
       ! m = rows of A and C, n = columns of B and C, k = cols of A / rows of B
       ! lda = m (leading dim of A), ldb = k (leading dim of B), ldc = m
       call backend_sgemm('N', 'N', m, num_samples, k, &
            1.0_real32, temp, m, b%val, k, 0.0_real32, c%val, m)
#else
       c%val = matmul(temp, b%val)
#endif
    elseif(.not.b%is_sample_dependent)then
       if(size(a%shape).ne.1)then
          write(err_msg,'("Matrix multiplication not implemented for array ''a'' &
               &rank: ",I0)') size(a%shape)
          call stop_program(err_msg)
          return
       end if
       ! C(n, S) = B^T(n, k) * A(k, S) where B is the weight matrix
       k = b%shape(1)
       n = b%shape(2)
       num_samples = size(a%val, 2)
       c => b%create_result(array_shape=[n, num_samples])
       temp(1:k, 1:n) => b%val
#ifdef USE_BLAS
       ! sgemm: C = alpha * op(A) * B + beta * C with transa='T'
       ! Computes C(n, S) = temp^T(n, k) * a%val(k, S)
       ! m = n (rows of result), n_arg = S, k = k (shared dim)
       ! lda = k (leading dim of temp before transpose), ldb = k, ldc = n
       call backend_sgemm('T', 'N', n, num_samples, k, &
            1.0_real32, temp, k, a%val, k, 0.0_real32, c%val, n)
#else
       c%val = matmul(transpose(temp), a%val)
#endif
    else
       write(0,*) "NOT SURE WHAT TO DO YET"
       stop 0
    end if

    c%is_sample_dependent = .true.
    c%get_partial_left => get_partial_matmul_left
    c%get_partial_right => get_partial_matmul_right
    c%get_partial_left_val => get_partial_matmul_left_val
    c%get_partial_right_val => get_partial_matmul_right_val
    c%get_partial_left_val_sum => get_partial_matmul_left_val_sum
    c%get_partial_right_val_sum => get_partial_matmul_right_val_sum
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'matmul'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function matmul_arrays
!-------------------------------------------------------------------------------
  module function matmul_real2d(a, b) result(c)
    !! Matrix multiplication of a real array and an autodiff array
    !! Computes C = a * b where a is autodiff (vector per sample) and b is a
    !! real 2D matrix. Equivalent to C(:,s) = b^T * a(:,s) for each sample s.
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    integer :: s, i, rows, cols, num_samples

    rows = size(b, 1)
    cols = size(b, 2)
    num_samples = size(a%val, 2)

    c => a%create_result(array_shape = [cols, num_samples])
#ifdef USE_BLAS
    ! C(cols, S) = b^T(cols, rows) * a%val(rows, S)
    ! sgemm: transa='T' transposes b in-place, no temporary needed
    ! m = cols, n = S, k = rows
    ! lda = rows (leading dim of b before transpose), ldb = rows, ldc = cols
    call backend_sgemm('T', 'N', cols, num_samples, rows, &
         1.0_real32, b, rows, a%val, rows, 0.0_real32, c%val, cols)
#else
    c%val = matmul(transpose(b), a%val)
#endif

    c%is_sample_dependent = a%is_sample_dependent
    c%get_partial_left => get_partial_matmul_left
    c%get_partial_right => get_partial_matmul_right
    c%get_partial_left_val => get_partial_matmul_left_val
    c%get_partial_right_val => get_partial_matmul_right_val
    c%get_partial_left_val_sum => get_partial_matmul_left_val_sum
    c%get_partial_right_val_sum => get_partial_matmul_right_val_sum
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'matmul_scalar'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
    allocate(b_array)
    b_array%is_sample_dependent = .false.
    b_array%shape = shape(b)
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[size(b,1), size(b,2), 1])
    b_array%val(:,1) = reshape(b, [size(b,1)*size(b,2)])
    c%right_operand => b_array
    c%owns_right_operand = .true.
  end function matmul_real2d
!-------------------------------------------------------------------------------
  module function real2d_matmul(a, b) result(c)
    !! Matrix multiplication of a real 2D matrix and an autodiff array
    !! Computes C = a * b where a is a real matrix and b is autodiff
    implicit none
    real(real32), dimension(:,:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c
    type(array_type), pointer :: a_array

    integer :: s, i, m, k, num_samples

    m = size(a, 1)
    k = size(a, 2)
    num_samples = size(b%val, 2)

    c => b%create_result(array_shape = [m, num_samples])
#ifdef USE_BLAS
    ! C(m, S) = a(m, k) * b%val(k, S)
    ! sgemm: standard C = alpha * A * B, no transposes
    ! m = m, n = S, k = k
    ! lda = m, ldb = k, ldc = m
    call backend_sgemm('N', 'N', m, num_samples, k, &
         1.0_real32, a, m, b%val, k, 0.0_real32, c%val, m)
#else
    c%val = matmul(a, b%val)
#endif

    c%is_sample_dependent = b%is_sample_dependent
    c%get_partial_left => get_partial_matmul_left
    c%get_partial_right => get_partial_matmul_right
    c%get_partial_left_val => get_partial_matmul_left_val
    c%get_partial_right_val => get_partial_matmul_right_val
    c%get_partial_left_val_sum => get_partial_matmul_left_val_sum
    c%get_partial_right_val_sum => get_partial_matmul_right_val_sum
    if(b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = b%is_forward
       c%operation = 'matmul_scalar'
       c%right_operand => b
       c%owns_right_operand = b%is_temporary
    end if
    allocate(a_array)
    a_array%is_sample_dependent = .false.
    a_array%shape = shape(a)
    a_array%requires_grad = .false.
    call a_array%allocate(array_shape=[size(a,1), size(a,2), 1])
    a_array%val(:,1) = reshape(a, [size(a,1)*size(a,2)])
    c%left_operand => a_array
    c%owns_left_operand = .true.
  end function real2d_matmul
!-------------------------------------------------------------------------------
  function get_partial_matmul_left(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand of matmul
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    logical :: right_is_temporary_local
    type(array_type), pointer :: ptr

    right_is_temporary_local = this%right_operand%is_temporary
    this%right_operand%is_temporary = .false.
    if(size(this%right_operand%shape).eq.2)then
       if(this%is_forward)then
          ptr => matmul( upstream_grad, this%right_operand )
       else
          ptr => matmul( upstream_grad, transpose(this%right_operand) )
       end if
    elseif(size(upstream_grad%shape).eq.2)then
       if(this%is_forward)then
          ptr => matmul( upstream_grad, this%right_operand )
       else
          ptr => matmul( transpose(upstream_grad), this%right_operand )
       end if
    else
       ptr => upstream_grad .outer. this%right_operand
    end if
    this%right_operand%is_temporary = right_is_temporary_local
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_matmul_left
!-------------------------------------------------------------------------------
  function get_partial_matmul_right(this, upstream_grad) result(output)
    !! Get partial derivative with respect to right operand of matmul
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    if(size(this%left_operand%shape).eq.2)then
       if(this%is_forward)then
          ptr => matmul(this%left_operand, upstream_grad)
       else
          ptr => matmul(transpose(this%left_operand), upstream_grad)
       end if
    elseif(size(upstream_grad%shape).eq.2)then
       if(this%is_forward)then
          ptr => matmul(this%left_operand, upstream_grad)
       else
          ptr => matmul(this%left_operand, transpose(upstream_grad))
       end if
    else
       ptr => this%left_operand .outer. upstream_grad
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)

  end function get_partial_matmul_right
!-------------------------------------------------------------------------------
#ifdef USE_BLAS
  subroutine get_partial_matmul_left_val(this, upstream_grad, output)
#else
  pure subroutine get_partial_matmul_left_val(this, upstream_grad, output)
#endif
    !! Compute gradient w.r.t. left operand for matmul in reverse mode.
    !!
    !! For C = A * B (where A is the left operand):
    !!   dL/dA = dL/dC * B^T  (2D case, equivalent to B * dL/dC in flat storage)
    !!   dL/dA = upstream_grad (x) right_operand  (outer product case)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, j, s, m, n, num_elements, num_batches, num_rhs

    num_batches = size(upstream_grad, 2)

    if(size(this%right_operand%shape).eq.2)then
       m = this%right_operand%shape(1)
       n = this%right_operand%shape(2)
       if(this%right_operand%is_sample_dependent)then
          ! Per-sample: weight matrix varies per sample.
          ! output(:,s) = W_s * upstream_grad(:,s) where W_s = reshape(right%val(:,s), [m,n])
          ! Equivalent to dL/dA_flat = dL/dC * B_s^T per sample.
          ! Uses intrinsic matmul — BLAS overhead not worthwhile for per-sample vectors.
          block
            real(real32), dimension(m, n) :: temp
            do s = 1, num_batches
               temp = reshape(this%right_operand%val(:,s), [m, n])
               output(:,s) = matmul(upstream_grad(:,s), transpose(temp))
            end do
          end block
       else
          ! Non-sample-dependent: single batch operation across all samples.
          ! output(m, S) = W(m, n) * upstream_grad(n, S)
          ! Mathematically: dL/dA = dL/dC * B^T for each sample, batched.
#ifdef USE_BLAS
          block
            real(real32), pointer :: W(:,:)
            ! Pointer view avoids reshape: right_operand%val(:,1) is column-major
            ! contiguous (m, n) data. No copy needed.
            W(1:m, 1:n) => this%right_operand%val(:,1)
            ! sgemm: C = alpha * A * B + beta * C
            ! A = W(m, n), B = upstream_grad(n, S), C = output(m, S)
            ! m_arg = m (rows of W and output)
            ! n_arg = num_batches (columns of upstream_grad and output)
            ! k = n (columns of W / rows of upstream_grad)
            ! lda = m (leading dim of W), ldb = n, ldc = m
            call backend_sgemm('N', 'N', m, num_batches, n, &
                 1.0_real32, W, m, upstream_grad, n, &
                 0.0_real32, output, m)
          end block
#else
          block
            real(real32), dimension(m, n) :: temp
            temp = reshape(this%right_operand%val(:,1), [m, n])
            output = matmul(temp, upstream_grad)
          end block
#endif
       end if
    else
       ! Outer product case: output(i + (j-1)*num_el, s) = grad(i,s) * right(j,s)
       num_elements = size(upstream_grad,1)
       num_rhs = size(this%right_operand%val,1)
       if(this%right_operand%is_sample_dependent)then
          do concurrent(s = 1:num_batches, j = 1:num_rhs)
             output((j-1)*num_elements+1:j*num_elements, s) = &
                  this%right_operand%val(j,s) * upstream_grad(:,s)
          end do
       else
          do j = 1, num_rhs
             output((j-1)*num_elements+1:j*num_elements, :) = &
                  this%right_operand%val(j, 1) * upstream_grad
          end do
       end if
    end if

  end subroutine get_partial_matmul_left_val
!-------------------------------------------------------------------------------
#ifdef USE_BLAS
  subroutine get_partial_matmul_right_val(this, upstream_grad, output)
#else
  pure subroutine get_partial_matmul_right_val(this, upstream_grad, output)
#endif
    !! Compute gradient w.r.t. right operand for matmul in reverse mode.
    !!
    !! For C = A * B (where B is the right operand):
    !!   dL/dB = A^T * dL/dC  (2D case)
    !!   dL/dB = left_operand (x) upstream_grad  (outer product case)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, j, s, m, n, num_elements, num_batches, num_upstream

    num_batches = size(upstream_grad, 2)

    if(size(this%left_operand%shape).eq.2)then
       m = this%left_operand%shape(1)
       n = this%left_operand%shape(2)
       if(this%left_operand%is_sample_dependent)then
          ! Per-sample: weight matrix varies per sample.
          ! output(:,s) = W_s^T * upstream_grad(:,s)
          ! where W_s = reshape(left%val(:,s), [m, n])
          ! Mathematically: dL/dB_flat = A_s^T * dL/dC per sample.
          ! Uses intrinsic matmul — BLAS overhead not worthwhile for per-sample vectors.
          block
            real(real32), dimension(m, n) :: temp
            do s = 1, num_batches
               temp = reshape(this%left_operand%val(:,s), [m, n])
               output(:,s) = matmul(transpose(temp), upstream_grad(:,s))
            end do
          end block
       else
          ! Non-sample-dependent: single batch operation across all samples.
          ! output(n, S) = W^T(n, m) * upstream_grad(m, S)
          ! Mathematically: dL/dB = A^T * dL/dC for each sample, batched.
#ifdef USE_BLAS
          block
            real(real32), pointer :: W(:,:)
            ! Pointer view avoids reshape and transpose: left_operand%val(:,1) is
            ! column-major contiguous (m, n) data. SGEMM transposes in-place.
            W(1:m, 1:n) => this%left_operand%val(:,1)
            ! sgemm: C = alpha * A^T * B + beta * C
            ! A = W(m, n), transposed to W^T(n, m); B = upstream_grad(m, S)
            ! m_arg = n (rows of W^T and output)
            ! n_arg = num_batches (columns of upstream_grad and output)
            ! k = m (columns of W^T / rows of upstream_grad)
            ! lda = m (leading dim of W before transpose), ldb = m, ldc = n
            call backend_sgemm('T', 'N', n, num_batches, m, &
                 1.0_real32, W, m, upstream_grad, m, &
                 0.0_real32, output, n)
          end block
#else
          block
            real(real32), dimension(n, m) :: temp_t
            temp_t = transpose(reshape(this%left_operand%val(:,1), [m, n]))
            output = matmul(temp_t, upstream_grad)
          end block
#endif
       end if
    else
       ! Outer product case: output(i + (j-1)*num_el, s) = left(i,s) * grad(j,s)
       num_elements = size(this%left_operand%val,1)
       num_upstream = size(upstream_grad, 1)
       if(this%left_operand%is_sample_dependent)then
          do concurrent(s = 1:num_batches, j = 1:num_upstream)
             output((j-1)*num_elements+1:j*num_elements, s) = &
                  upstream_grad(j,s) * this%left_operand%val(:,s)
          end do
       else
          do concurrent(s=1:num_batches, j=1:num_upstream)
             output((j-1)*num_elements+1:j*num_elements, s) = &
                  this%left_operand%val(:, 1) * upstream_grad(j, s)
          end do
       end if
    end if

  end subroutine get_partial_matmul_right_val
!###############################################################################


!###############################################################################
! Sum-reduced gradient functions for matmul
! These compute sum(partial(upstream_grad), dim=2) directly, avoiding
! the large (n_elem, num_samples) intermediate array allocation.
!###############################################################################
#ifdef USE_BLAS
  subroutine get_partial_matmul_left_val_sum(this, upstream_grad, output)
#else
  pure subroutine get_partial_matmul_left_val_sum(this, upstream_grad, output)
#endif
    !! Sum-reduced gradient w.r.t. left operand for matmul.
    !! For the outer product case (rank-1 right operand), this computes:
    !!   output = sum_s(upstream(:,s) (x) right(:,s)) = upstream * right^T
    !! using a single SGEMM call instead of computing the full (n_elem, S) array.
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    integer :: i, j, s, m, n, num_elements, num_batches, num_rhs

    num_batches = size(upstream_grad, 2)

    if(size(this%right_operand%shape).eq.2)then
       ! 2D case: output = sum_s(W * upstream(:,s)) = W * sum(upstream, dim=2)
       m = this%right_operand%shape(1)
       n = this%right_operand%shape(2)
       if(this%right_operand%is_sample_dependent)then
          ! Per-sample weight matrices: fall back to explicit loop
          block
            real(real32), dimension(m, n) :: temp
            real(real32), dimension(m) :: row_result
            output = 0.0_real32
            do s = 1, num_batches
               temp = reshape(this%right_operand%val(:,s), [m, n])
               row_result = matmul(upstream_grad(:,s), transpose(temp))
               output = output + row_result
            end do
          end block
       else
          ! Non-sample-dependent: W * sum(upstream, dim=2) via matvec
#ifdef USE_BLAS
          block
            real(real32), pointer :: W(:,:)
            real(real32), dimension(size(upstream_grad,1)) :: upstream_sum
            W(1:m, 1:n) => this%right_operand%val(:,1)
            upstream_sum = sum(upstream_grad, dim=2)
            call backend_sgemv('N', m, n, 1.0_real32, W, m, upstream_sum, 1, &
                 0.0_real32, output, 1)
          end block
#else
          block
            real(real32), dimension(m, n) :: temp
            real(real32), dimension(size(upstream_grad,1)) :: upstream_sum
            temp = reshape(this%right_operand%val(:,1), [m, n])
            upstream_sum = sum(upstream_grad, dim=2)
            output = matmul(temp, upstream_sum)
          end block
#endif
       end if
    else
       ! Outer product case: sum_s(upstream(i,s) * right(j,s))
       ! = matmul(upstream, right^T) stored column-major
       num_elements = size(upstream_grad, 1)
       num_rhs = size(this%right_operand%val, 1)
       if(this%right_operand%is_sample_dependent)then
#ifdef USE_BLAS
          ! SGEMM: result(num_elements, num_rhs) = upstream * right^T
          call backend_sgemm('N', 'T', num_elements, num_rhs, num_batches, &
               1.0_real32, upstream_grad, num_elements, &
               this%right_operand%val, num_rhs, &
               0.0_real32, output, num_elements)
#else
          ! Manual fused outer product + sum
          output = 0.0_real32
          do s = 1, num_batches
             do j = 1, num_rhs
                do i = 1, num_elements
                   output((j-1)*num_elements + i) = &
                        output((j-1)*num_elements + i) + &
                        this%right_operand%val(j,s) * upstream_grad(i,s)
                end do
             end do
          end do
#endif
       else
          ! Right operand not sample-dependent: outer product with scalar right
          ! sum_s(right(j,1) * upstream(i,s)) = right(j,1) * sum_s(upstream(i,s))
          block
            real(real32), dimension(num_elements) :: upstream_sum
            upstream_sum = sum(upstream_grad, dim=2)
            do j = 1, num_rhs
               output((j-1)*num_elements+1:j*num_elements) = &
                    this%right_operand%val(j, 1) * upstream_sum
            end do
          end block
       end if
    end if

  end subroutine get_partial_matmul_left_val_sum
!###############################################################################


!###############################################################################
#ifdef USE_BLAS
  subroutine get_partial_matmul_right_val_sum(this, upstream_grad, output)
#else
  pure subroutine get_partial_matmul_right_val_sum(this, upstream_grad, output)
#endif
    !! Sum-reduced gradient w.r.t. right operand for matmul.
    !! For the outer product case (rank-1 left operand), this computes:
    !!   output = sum_s(left(:,s) (x) upstream(:,s)) = left * upstream^T
    !! using a single SGEMM call.
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:), intent(out) :: output

    integer :: i, j, s, m, n, num_elements, num_batches, num_upstream

    num_batches = size(upstream_grad, 2)

    if(size(this%left_operand%shape).eq.2)then
       ! 2D case: output = sum_s(W^T * upstream(:,s)) = W^T * sum(upstream, dim=2)
       m = this%left_operand%shape(1)
       n = this%left_operand%shape(2)
       if(this%left_operand%is_sample_dependent)then
          block
            real(real32), dimension(m, n) :: temp
            real(real32), dimension(n) :: col_result
            output = 0.0_real32
            do s = 1, num_batches
               temp = reshape(this%left_operand%val(:,s), [m, n])
               col_result = matmul(transpose(temp), upstream_grad(:,s))
               output = output + col_result
            end do
          end block
       else
#ifdef USE_BLAS
          block
            real(real32), pointer :: W(:,:)
            real(real32), dimension(size(upstream_grad,1)) :: upstream_sum
            W(1:m, 1:n) => this%left_operand%val(:,1)
            upstream_sum = sum(upstream_grad, dim=2)
            ! W^T * upstream_sum: sgemv with 'T'
            call backend_sgemv('T', m, n, 1.0_real32, W, m, upstream_sum, 1, &
                 0.0_real32, output, 1)
          end block
#else
          block
            real(real32), dimension(n, m) :: temp_t
            real(real32), dimension(size(upstream_grad,1)) :: upstream_sum
            temp_t = transpose(reshape(this%left_operand%val(:,1), [m, n]))
            upstream_sum = sum(upstream_grad, dim=2)
            output = matmul(temp_t, upstream_sum)
          end block
#endif
       end if
    else
       ! Outer product case: sum_s(left(i,s) * upstream(j,s))
       ! = matmul(left, upstream^T) stored column-major
       num_elements = size(this%left_operand%val, 1)
       num_upstream = size(upstream_grad, 1)
       if(this%left_operand%is_sample_dependent)then
#ifdef USE_BLAS
          call backend_sgemm('N', 'T', num_elements, num_upstream, num_batches, &
               1.0_real32, this%left_operand%val, num_elements, &
               upstream_grad, num_upstream, &
               0.0_real32, output, num_elements)
#else
          output = 0.0_real32
          do s = 1, num_batches
             do j = 1, num_upstream
                do i = 1, num_elements
                   output((j-1)*num_elements + i) = &
                        output((j-1)*num_elements + i) + &
                        upstream_grad(j,s) * this%left_operand%val(i,s)
                end do
             end do
          end do
#endif
       else
          block
            real(real32), dimension(num_upstream) :: upstream_sum
            upstream_sum = sum(upstream_grad, dim=2)
            do j = 1, num_upstream
               output((j-1)*num_elements+1:j*num_elements) = &
                    this%left_operand%val(:, 1) * upstream_sum(j)
            end do
          end block
       end if
    end if

  end subroutine get_partial_matmul_right_val_sum
!###############################################################################


!###############################################################################
  module function outer_product_arrays(a, b) result(c)
    !! Outer product of two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: i, j, s

    ! check shapes
    if(size(a%shape).ne.1 .or. size(b%shape).ne.1)then
       call stop_program("dot_product_arrays: only 1D arrays supported")
    elseif(size(a%val,2).ne.size(b%val,2))then
       call stop_program("dot_product_arrays: array length mismatch")
    end if

    c => a%create_result(array_shape = [size(a%val,1), size(b%val,1), size(a%val,2)])
    ! outer product 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       do concurrent(i=1:size(a%val,1), j=1:size(b%val,1))
          c%val(i + (j-1)*size(a%val,1),s) = a%val(i,s) * b%val(j,s)
       end do
    end do

    c%get_partial_left => get_partial_outer_product_left
    c%get_partial_right => get_partial_outer_product_right
    c%is_sample_dependent = a%is_sample_dependent
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'outer_product'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function outer_product_arrays
!-------------------------------------------------------------------------------
  function get_partial_outer_product_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: right_is_temporary_local
    type(array_type), pointer :: ptr

    right_is_temporary_local = this%right_operand%is_temporary
    this%right_operand%is_temporary = .false.
    if(this%is_forward)then
       ptr => upstream_grad .outer. this%right_operand
    else
       ptr => matmul(upstream_grad, this%right_operand)
    end if
    this%right_operand%is_temporary = right_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_outer_product_left
!-------------------------------------------------------------------------------
  function get_partial_outer_product_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    if(this%is_forward)then
       ptr => this%left_operand .outer. upstream_grad
    else
       ! mathematically should be ptr => transpose(upstream_grad) .mmul. this%left_operand
       ! but for how we store vectors, this SHOULD BE equivalent
       ptr => matmul(this%left_operand, upstream_grad)
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_outer_product_right
!###############################################################################


!###############################################################################
  module function dot_product_arrays(a, b) result(c)
    !! Dot product of two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s

    ! check shapes
    if(size(a%shape).ne.1 .or. size(b%shape).ne.1)then
       call stop_program("dot_product_arrays: only 1D arrays supported")
    elseif(any(shape(a%val).ne.shape(b%val)))then
       call stop_program("dot_product_arrays: array length mismatch")
    end if

    c => a%create_result(array_shape = [1, size(a%val,2)])
    do concurrent(s=1:size(a%val,2))
       c%val(1,s) = dot_product(a%val(:,s), b%val(:,s))
    end do

    c%get_partial_left => get_partial_dot_product_left
    c%get_partial_right => get_partial_dot_product_right
    c%get_partial_left_val => get_partial_dot_product_left_val
    c%get_partial_right_val => get_partial_dot_product_right_val
    c%is_sample_dependent = a%is_sample_dependent
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'dot_product'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function dot_product_arrays
!-------------------------------------------------------------------------------
  function get_partial_dot_product_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: right_is_temporary_local
    type(array_type), pointer :: ptr

    right_is_temporary_local = this%right_operand%is_temporary
    this%right_operand%is_temporary = .false.
    if(this%is_forward)then
       ptr => dot_product(upstream_grad, this%right_operand)
    else
       ptr => upstream_grad * this%right_operand
    end if
    this%right_operand%is_temporary = right_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_dot_product_left
!-------------------------------------------------------------------------------
  function get_partial_dot_product_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    if(this%is_forward)then
       ptr => dot_product(this%left_operand, upstream_grad)
    else
       ptr => upstream_grad * this%left_operand
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_dot_product_right
!-------------------------------------------------------------------------------
  pure subroutine get_partial_dot_product_left_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s

    do concurrent(s=1:size(upstream_grad,2), i=1:size(this%right_operand%val,1))
       output(i,s) = upstream_grad(1,s) * this%right_operand%val(i,s)
    end do
  end subroutine get_partial_dot_product_left_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_dot_product_right_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s

    do concurrent(s=1:size(upstream_grad,2), i=1:size(this%left_operand%val,1))
       output(i,s) = upstream_grad(1,s) * this%left_operand%val(i,s)
    end do
  end subroutine get_partial_dot_product_right_val
!###############################################################################


!###############################################################################
  module function transpose_array(a) result(c)
    !! Transpose an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    integer :: i, j, s

    if(size(a%shape) .ne. 2)then
       write(*,*) "ashape", a%shape
       call stop_program("transpose_array: only 2D arrays can be transposed")
    end if
    c => a%create_result(array_shape=[a%shape(2), a%shape(1), size(a%val,2)])
    ! transpose 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       do concurrent(i=1:a%shape(1), j=1:a%shape(2))
          c%val( (i-1)*a%shape(2) + j, s) = a%val( (j-1)*a%shape(1) + i, s)
       end do
    end do

    c%get_partial_left => get_partial_transpose_left
    ! c%get_partial_right => get_partial_transpose_right
    c%is_sample_dependent = a%is_sample_dependent
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'transpose'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function transpose_array
!-------------------------------------------------------------------------------
  function get_partial_transpose_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => transpose(upstream_grad)
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_transpose_left
! !-------------------------------------------------------------------------------
! !   function get_partial_transpose_right(this, upstream_grad) result(output)
! !     class(array_type), intent(inout) :: this
! !     type(array_type), intent(in) :: upstream_grad
! !     type(array_type) :: output

! !     output = transpose(this%left_operand)

! !   end function get_partial_transpose_right
!###############################################################################

end submodule diffstruc__operations_linalg_sub
