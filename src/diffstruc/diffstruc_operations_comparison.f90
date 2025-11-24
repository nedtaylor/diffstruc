module diffstruc__operations_comparison
  !! This module contains comparison operations for the diffstruc library.
  use coreutils, only: real32, stop_program
  use diffstruc__types, only: array_type, get_partial, &
       operator(+), operator(-), operator(*), operator(**)
  implicit none


  private

  public :: operator(.lt.), operator(.gt.), operator(.le.), operator(.ge.)
  public :: merge

  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface operator(.lt.)
     module procedure lt_scalar
  end interface

  interface operator(.le.)
     module procedure le_scalar
  end interface

  interface operator(.gt.)
     module procedure gt_scalar
  end interface

  interface operator(.ge.)
     module procedure ge_scalar
  end interface

  interface merge
     module procedure merge_array
     module procedure merge_scalar
     module procedure scalar_merge
     module procedure merge_real2d
  end interface


contains

!###############################################################################
  function lt_scalar(a, b) result(c)
    !! Less than comparison between autodiff array and scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    logical, dimension(size(a%val,1), size(a%val,2)) :: c

    c = a%val .lt. b

  end function lt_scalar
!###############################################################################


!###############################################################################
  function le_scalar(a, b) result(c)
    !! Less than or equal tocomparison between autodiff array and scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    logical, dimension(size(a%val,1), size(a%val,2)) :: c

    c = a%val .le. b

  end function le_scalar
!###############################################################################


!###############################################################################
  function gt_scalar(a, b) result(c)
    !! Greater than comparison between autodiff array and scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    logical, dimension(size(a%val,1), size(a%val,2)) :: c

    c = a%val .gt. b

  end function gt_scalar
!###############################################################################


!###############################################################################
  function ge_scalar(a, b) result(c)
    !! Greater than or equal to comparison between autodiff array and scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    logical, dimension(size(a%val,1), size(a%val,2)) :: c

    c = a%val .ge. b

  end function ge_scalar
!###############################################################################


!###############################################################################
  function merge_array(tsource, fsource, mask) result(c)
    !! Merge two autodiff arrays based on a mask
    implicit none
    class(array_type), intent(in), target :: tsource
    class(array_type), intent(in), target :: fsource
    logical, dimension(:,:), intent(in) :: mask
    type(array_type), pointer :: c

    integer :: i, j

    c => tsource%create_result(array_shape=[size(tsource%val,1), size(tsource%val,2)])
    ! merge 1D array by using shape to swap dimensions
    do concurrent(i=1:size(tsource%val,1), j=1:size(tsource%val,2))
       if(mask(i,j)) then
          c%val(i,j) = tsource%val(i,j)
       else
          c%val(i,j) = fsource%val(i,j)
       end if
    end do
    c%mask = mask

    c%get_partial_left => get_partial_merge_left
    c%get_partial_right => get_partial_merge_right
    c%get_partial_left_val => get_partial_merge_left_val
    c%get_partial_right_val => get_partial_merge_right_val
    if(tsource%requires_grad.or. fsource%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = tsource%is_forward .or. fsource%is_forward
       c%operation = 'merge'
       c%left_operand => tsource
       c%right_operand => fsource
       c%owns_left_operand = tsource%is_temporary
       c%owns_right_operand = fsource%is_temporary
    end if
  end function merge_array
!-------------------------------------------------------------------------------
  function merge_scalar(tsource, fsource, mask) result(c)
    !! Merge two autodiff arrays based on a mask
    implicit none
    class(array_type), intent(in), target :: tsource
    real(real32), intent(in) :: fsource
    logical, dimension(:,:), intent(in) :: mask
    type(array_type), pointer :: c

    integer :: i, j

    if(size(tsource%shape) .ne. 1)then
       call stop_program("merge_array: only 1D arrays can be merged")
    end if

    c => tsource%create_result(array_shape=[size(tsource%val,1), size(tsource%val,2)])
    ! merge 1D array by using shape to swap dimensions
    do concurrent(i=1:size(tsource%val,1), j=1:size(tsource%val,2))
       if(mask(i,j)) then
          c%val(i,j) = tsource%val(i,j)
       else
          c%val(i,j) = fsource
       end if
    end do
    c%mask = mask

    c%get_partial_left => get_partial_merge_left
    c%get_partial_left_val => get_partial_merge_left_val
    if(tsource%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = tsource%is_forward
       c%operation = 'merge_scalar'
       c%left_operand => tsource
       c%owns_left_operand = tsource%is_temporary
    end if
  end function merge_scalar
!-------------------------------------------------------------------------------
  function scalar_merge(tsource, fsource, mask) result(c)
    !! Merge two autodiff arrays based on a mask
    implicit none
    real(real32), intent(in) :: tsource
    class(array_type), intent(in), target :: fsource
    logical, dimension(:,:), intent(in) :: mask
    type(array_type), pointer :: c

    c => merge_scalar(fsource, tsource, .not.mask)
  end function scalar_merge
!-------------------------------------------------------------------------------
  function merge_real2d(tsource, fsource, mask) result(c)
    !! Merge two autodiff arrays based on a mask
    implicit none
    class(array_type), intent(in), target :: tsource
    real(real32), dimension(:,:), intent(in) :: fsource
    logical, dimension(:,:), intent(in) :: mask
    type(array_type), pointer :: c

    integer :: i, j !, itmp1
    !  integer, dimension(:,:), allocatable :: adj_ja_tmp

    if(allocated(tsource%shape))then
       if(size(tsource%shape) .ne. 1)then
          call stop_program("merge_array: only 1D arrays can be merged")
       end if
    end if

    allocate(c)
    call c%allocate(array_shape=[size(tsource%val,1), size(tsource%val,2)])
    ! merge 1D array by using shape to swap dimensions
    !  allocate(adj_ja_tmp(1, size(mask)))
    !  itmp1 = 0
    do concurrent( i = 1: size(tsource%val,1), j = 1: size(tsource%val,2))
       if(mask(i,j)) then
          c%val(i,j) = tsource%val(i,j)
          !  if(.not.allocated(c%indices))then
          !    c%indices = [i]
          !  elseif(c%indices(size(c%indices)) .ne. i) then
          !    c%indices = [c%indices, i]
          !  end if
          !  itmp1 = itmp1 + 1
          !  adj_ja_tmp(1,itmp1) = j
       else
          c%val(i,j) = fsource(i,j)
       end if
    end do
    c%mask = mask
    !  allocate(c%adj_ja(1, itmp1))
    !  c%adj_ja(1,:) = adj_ja_tmp(1,1:itmp1)


    c%get_partial_left => get_partial_merge_left
    c%get_partial_left_val => get_partial_merge_left_val
    if(tsource%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = tsource%is_forward
       c%operation = 'merge'
       c%left_operand => tsource
       c%owns_left_operand = tsource%is_temporary
    end if
  end function merge_real2d
!-------------------------------------------------------------------------------
  function get_partial_merge_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => merge(upstream_grad, 0._real32, this%mask)
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_merge_left
!-------------------------------------------------------------------------------
  function get_partial_merge_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => merge(0._real32, upstream_grad, this%mask)
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_merge_right
!-------------------------------------------------------------------------------
  subroutine get_partial_merge_left_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, j

    do concurrent(i = 1:size(this%val,1), j = 1:size(this%val,2))
       if(this%mask(i,j)) then
          output(i,j) = upstream_grad(i,j)
       else
          output(i,j) = 0._real32
       end if
    end do

  end subroutine get_partial_merge_left_val
!-------------------------------------------------------------------------------
  subroutine get_partial_merge_right_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(inout) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, j

    do concurrent(i = 1:size(this%val,1), j = 1:size(this%val,2))
       if(this%mask(i,j)) then
          output(i,j) = 0._real32
       else
          output(i,j) = upstream_grad(i,j)
       end if
    end do

  end subroutine get_partial_merge_right_val
!###############################################################################

end module diffstruc__operations_comparison
