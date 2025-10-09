module diffstruc__operations_comparison
  !! This module contains comparison operations for the diffstruc library.
  use corestruc, only: real32, stop_program
  use diffstruc__types, only: array_type, get_partial, &
       operator(+), operator(-), operator(*), operator(**)
  implicit none


  private

  public :: operator(.lt.), operator(.gt.)
  public :: merge

  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface operator(.lt.)
     module procedure lt_scalar
  end interface

  interface operator(.gt.)
     module procedure gt_scalar
  end interface

  interface merge
     module procedure merge_scalar
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
  function merge_scalar(tsource, fsource, mask) result(c)
    !! Merge two autodiff arrays based on a mask
    implicit none
    class(array_type), intent(in), target :: tsource
    real(real32), intent(in) :: fsource
    logical, dimension(:,:), intent(in) :: mask
    type(array_type), pointer :: c

    integer :: i, j, s

    if(size(tsource%shape) .ne. 1)then
       call stop_program("merge_array: only 1D arrays can be merged")
    end if

    allocate(c)
    call c%allocate(array_shape=[size(tsource%val,1), size(tsource%val,2)])
    ! merge 1D array by using shape to swap dimensions
    do concurrent(s=1:size(tsource%val,2))
       do concurrent(i=1:size(tsource%val,1), j=1:size(tsource%val,2))
          if(mask(i,j)) then
             c%val(i,j) = tsource%val(i,j)
          else
             c%val(i,j) = fsource
          end if
       end do
    end do
    c%mask = mask

    c%get_partial_left => get_partial_merge
    if(tsource%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = tsource%is_forward
       c%is_leaf = .false.
       c%operation = 'merge'
       c%left_operand => tsource
    end if
  end function merge_scalar
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


    c%get_partial_left => get_partial_merge
    if(tsource%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = tsource%is_forward
       c%is_leaf = .false.
       c%operation = 'merge'
       c%left_operand => tsource
    end if
  end function merge_real2d
!-------------------------------------------------------------------------------
  function get_partial_merge(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = merge(upstream_grad, 0._real32, this%mask)

  end function get_partial_merge
!###############################################################################

end module diffstruc__operations_comparison
