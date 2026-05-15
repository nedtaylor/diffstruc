module diffstruc__backend_linalg
  !! Internal backend dispatch for dense BLAS-like kernels.
  use coreutils, only: real32
  use iso_c_binding, only: c_char, c_float, c_int
  implicit none

  private

  integer, parameter, public :: diffstruc_backend_auto = 0
  integer, parameter, public :: diffstruc_backend_legacy = 1
  integer, parameter, public :: diffstruc_backend_accelerate = 2
  integer, parameter, public :: diffstruc_backend_metal = 3

  integer, save :: backend_override = -1
  logical, save :: backend_override_active = .false.

  public :: set_backend_mode
  public :: reset_backend_mode
  public :: get_backend_mode
  public :: backend_name
  public :: backend_available
  public :: resolve_gemm_backend
#ifdef USE_BLAS
  public :: backend_sgemm
  public :: backend_sgemv

  interface
     subroutine sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
       import :: real32
       character(len=1), intent(in) :: transa, transb
       integer, intent(in) :: m, n, k, lda, ldb, ldc
       real(real32), intent(in) :: alpha, beta
       real(real32), intent(in) :: a(lda,*), b(ldb,*)
       real(real32), intent(inout) :: c(ldc,*)
     end subroutine sgemm

     subroutine sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
       import :: real32
       character(len=1), intent(in) :: trans
       integer, intent(in) :: m, n, lda, incx, incy
       real(real32), intent(in) :: alpha, beta
       real(real32), intent(in) :: a(lda,*), x(*)
       real(real32), intent(inout) :: y(*)
     end subroutine sgemv
  end interface
#endif

  interface
     integer(c_int) function diffstruc_apple_backend_available(backend) &
          bind(C, name='diffstruc_apple_backend_available')
       import :: c_int
       integer(c_int), value :: backend
     end function diffstruc_apple_backend_available

     integer(c_int) function diffstruc_apple_resolve_backend( &
          backend, m, n, k, is_gemv &
     ) bind(C, name='diffstruc_apple_resolve_backend')
       import :: c_int
       integer(c_int), value :: backend, m, n, k, is_gemv
     end function diffstruc_apple_resolve_backend

#ifdef USE_BLAS
     integer(c_int) function diffstruc_apple_sgemm(backend, transa, transb, m, n, k, &
          alpha, a, lda, b, ldb, beta, c, ldc) bind(C, name='diffstruc_apple_sgemm')
       import :: c_char, c_float, c_int
       integer(c_int), value :: backend, m, n, k, lda, ldb, ldc
       character(kind=c_char), value :: transa, transb
       real(c_float), value :: alpha, beta
       real(c_float), intent(in) :: a(lda,*), b(ldb,*)
       real(c_float), intent(inout) :: c(ldc,*)
     end function diffstruc_apple_sgemm

     integer(c_int) function diffstruc_apple_sgemv(backend, trans, m, n, alpha, &
          a, lda, x, incx, beta, y, incy) bind(C, name='diffstruc_apple_sgemv')
       import :: c_char, c_float, c_int
       integer(c_int), value :: backend, m, n, lda, incx, incy
       character(kind=c_char), value :: trans
       real(c_float), value :: alpha, beta
       real(c_float), intent(in) :: a(lda,*), x(*)
       real(c_float), intent(inout) :: y(*)
     end function diffstruc_apple_sgemv
#endif
  end interface

contains

  subroutine set_backend_mode(mode)
    implicit none
    integer, intent(in) :: mode

    backend_override = mode
    backend_override_active = .true.
  end subroutine set_backend_mode

  subroutine reset_backend_mode()
    implicit none

    backend_override = -1
    backend_override_active = .false.
  end subroutine reset_backend_mode

  integer function get_backend_mode() result(mode)
    implicit none
    character(len=64) :: env_value
    integer :: env_status, env_length

    if(backend_override_active)then
       mode = backend_override
       return
    end if

    mode = diffstruc_backend_auto
    env_value = ''
    call get_environment_variable('DIFFSTRUC_LINALG_BACKEND', env_value, &
         length=env_length, status=env_status)
    if(env_status.eq.0) mode = parse_backend_name(env_value(:env_length))
  end function get_backend_mode

  character(len=16) function backend_name(mode) result(name)
    implicit none
    integer, intent(in) :: mode

    select case(mode)
    case(diffstruc_backend_auto)
       name = 'auto'
    case(diffstruc_backend_legacy)
       name = 'legacy'
    case(diffstruc_backend_accelerate)
       name = 'accelerate'
    case(diffstruc_backend_metal)
       name = 'metal'
    case default
       name = 'unknown'
    end select
  end function backend_name

  logical function backend_available(mode) result(is_available)
    implicit none
    integer, intent(in) :: mode

    is_available = .false.
    select case(mode)
    case(diffstruc_backend_auto)
       is_available = .true.
    case(diffstruc_backend_legacy)
#ifdef USE_BLAS
       is_available = .true.
#endif
    case(diffstruc_backend_accelerate)
       is_available = diffstruc_apple_backend_available(int(mode, c_int)).eq.1_c_int
    case(diffstruc_backend_metal)
#ifdef DIFFSTRUC_ENABLE_METAL_BACKEND
       is_available = diffstruc_apple_backend_available(int(mode, c_int)).eq.1_c_int
#endif
    end select
  end function backend_available

  integer function resolve_gemm_backend(requested_mode, m, n, k, is_gemv) result(mode)
    implicit none
    integer, intent(in) :: requested_mode, m, n, k
    logical, intent(in) :: is_gemv
    integer :: requested_apple
    integer(c_int) :: resolved

    mode = requested_mode
    if(requested_mode.eq.diffstruc_backend_legacy) return

    select case(requested_mode)
    case(diffstruc_backend_auto)
#ifdef DIFFSTRUC_ENABLE_METAL_BACKEND
       requested_apple = diffstruc_backend_auto
#else
       requested_apple = diffstruc_backend_accelerate
#endif
    case(diffstruc_backend_accelerate)
       requested_apple = diffstruc_backend_accelerate
    case(diffstruc_backend_metal)
#ifdef DIFFSTRUC_ENABLE_METAL_BACKEND
       requested_apple = diffstruc_backend_metal
#else
       requested_apple = diffstruc_backend_accelerate
#endif
    case default
       requested_apple = diffstruc_backend_accelerate
    end select

    resolved = diffstruc_apple_resolve_backend( &
         int(requested_apple, c_int), int(m, c_int), &
         int(n, c_int), int(k, c_int), merge(1_c_int, 0_c_int, is_gemv) &
    )
    if(resolved.gt.0_c_int)then
       mode = resolved
       return
    end if

#ifdef USE_BLAS
    mode = diffstruc_backend_legacy
#else
    mode = diffstruc_backend_auto
#endif
  end function resolve_gemm_backend

#ifdef USE_BLAS
  subroutine backend_sgemm( &
       transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc &
  )
    implicit none
    character(len=1), intent(in) :: transa, transb
    integer, intent(in) :: m, n, k, lda, ldb, ldc
    real(real32), intent(in) :: alpha, beta
    real(real32), intent(in) :: a(lda,*), b(ldb,*)
    real(real32), intent(inout) :: c(ldc,*)

    character(kind=c_char, len=1) :: transa_c, transb_c
    integer :: mode
    integer(c_int) :: status

    mode = resolve_gemm_backend(get_backend_mode(), m, n, k, .false.)
    if(mode.eq.diffstruc_backend_accelerate .or. mode.eq.diffstruc_backend_metal)then
       transa_c = char(iachar(transa), kind=c_char)
       transb_c = char(iachar(transb), kind=c_char)
       status = diffstruc_apple_sgemm( &
            int(mode, c_int), transa_c, transb_c, int(m, c_int), &
            int(n, c_int), int(k, c_int), &
            real(alpha, c_float), a, int(lda, c_int), b, &
            int(ldb, c_int), real(beta, c_float), c, int(ldc, c_int))
       if(status.eq.0_c_int) return
    end if

    call sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
  end subroutine backend_sgemm

  subroutine backend_sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
    implicit none
    character(len=1), intent(in) :: trans
    integer, intent(in) :: m, n, lda, incx, incy
    real(real32), intent(in) :: alpha, beta
    real(real32), intent(in) :: a(lda,*), x(*)
    real(real32), intent(inout) :: y(*)

    character(kind=c_char, len=1) :: trans_c
    integer :: mode
    integer(c_int) :: status

    mode = resolve_gemm_backend(get_backend_mode(), m, n, 1, .true.)
    if(mode.eq.diffstruc_backend_accelerate .or. mode.eq.diffstruc_backend_metal)then
       trans_c = char(iachar(trans), kind=c_char)
       status = diffstruc_apple_sgemv( &
            int(mode, c_int), trans_c, int(m, c_int), int(n, c_int), &
            real(alpha, c_float), a, int(lda, c_int), x, int(incx, c_int), &
            real(beta, c_float), y, int(incy, c_int))
       if(status.eq.0_c_int) return
    end if

    call sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
  end subroutine backend_sgemv
#endif

  integer function parse_backend_name(raw_value) result(mode)
    implicit none
    character(len=*), intent(in) :: raw_value
    character(len=len(raw_value)) :: value
    integer :: i

    value = adjustl(raw_value)
    do i = 1, len(value)
       if(value(i:i).ge.'A' .and. value(i:i).le.'Z') &
            value(i:i) = achar(iachar(value(i:i)) + 32)
    end do

    select case(trim(value))
    case('legacy', 'blas')
       mode = diffstruc_backend_legacy
    case('accelerate')
       mode = diffstruc_backend_accelerate
    case('metal')
       mode = diffstruc_backend_metal
    case default
       mode = diffstruc_backend_auto
    end select
  end function parse_backend_name

end module diffstruc__backend_linalg
