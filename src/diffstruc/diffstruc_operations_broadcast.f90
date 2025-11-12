module diffstruc__operations_broadcast
  !! This module contains broadcast operations for the diffstruc library.
  use coreutils, only: real32
  use diffstruc__types, only: array_type, get_partial, &
       operator(+), operator(-), operator(*), operator(**)
  implicit none


  private

  public :: operator(.concat.), operator(.ltrim.), operator(.rtrim.), &
       operator(.index.), reverse_index, &
       pack, unpack

  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface operator(.concat.)
     module procedure concat_arrays
  end interface

  interface operator(.ltrim.)
     module procedure ltrim_array
  end interface

  interface operator(.rtrim.)
     module procedure rtrim_array
  end interface

  interface operator(.index.)
     module procedure index_array
  end interface

  interface reverse_index
     module procedure reverse_index_array
  end interface

  interface pack
     module procedure pack_array
  end interface

  interface unpack
     module procedure unpack_array
  end interface


contains

!###############################################################################
  function concat_arrays(a, b) result(c)
    !! Concatenate two autodiff arrays along the first dimension
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: i, j, s

    c => a%create_result(array_shape = [size(a%val,1) + size(b%val,1), size(a%val,2)])
    ! concatenate 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       do concurrent(i=1:1, j=1:size(a%val,1))
          c%val( i, s) = a%val( i, s)
       end do
       do concurrent(i=1:1, j=1:size(b%val,1))
          c%val( size(a%val,1) + i, s) = b%val( i, s)
       end do
    end do

    c%get_partial_left => get_partial_concat_left
    c%get_partial_right => get_partial_concat_right
    if(a%requires_grad .or. b%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'concat'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function concat_arrays
!-------------------------------------------------------------------------------
  function get_partial_concat_left(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad .ltrim. this%left_operand%shape(1)

  end function get_partial_concat_left
!-------------------------------------------------------------------------------
  function get_partial_concat_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => upstream_grad .rtrim. this%right_operand%shape(1)
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_concat_right
!###############################################################################


!###############################################################################
  function ltrim_array(a, b) result(c)
    !! Left trim an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: b
    type(array_type), pointer :: c

    integer :: i, j, s

    c => a%create_result(array_shape = [b, size(a%val,2)])
    ! left trim 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       c%val( :, s) = a%val( 1:b, s)
    end do

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'ltrim'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function ltrim_array
!###############################################################################


!###############################################################################
  function rtrim_array(a, b) result(c)
    !! Right trim an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: b
    type(array_type), pointer :: c

    integer :: i, j, s

    c => a%create_result(array_shape = [b, size(a%val,2)])
    ! right trim 1D array by using shape to swap dimensions
    do concurrent(s=1:size(a%val,2))
       c%val( :, s) = a%val( size(a%val,1)-b+1:size(a%val,1), s)
    end do

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'rtrim'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function rtrim_array
!###############################################################################


!###############################################################################
  function index_array(a, indices) result(c)
    !! Index an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, dimension(:), intent(in) :: indices
    type(array_type), pointer :: c

    integer :: i, s

    allocate(c)
    call c%allocate(array_shape=[size(a%val,1), size(indices)])
    do concurrent(s=1:size(indices), i=1:size(a%val,1))
       c%val(i, s) = a%val(i, indices(s))
    end do
    c%indices = indices

    c%get_partial_left => get_partial_index
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'index'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function index_array
!-------------------------------------------------------------------------------
  function get_partial_index(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => reverse_index( &
         upstream_grad, indices=this%indices, from=.false., &
         new_index_size=size(this%left_operand%val, 2) &
    )
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_index
!###############################################################################


!###############################################################################
  function reverse_index_array(a, indices, from, new_index_size) result(c)
    !! Index an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, dimension(:), intent(in) :: indices
    logical, intent(in) :: from
    integer, intent(in) :: new_index_size
    type(array_type), pointer :: c

    integer :: i, s

    allocate(c)
    if(from) then
       call c%allocate(array_shape=[size(a%val,1), new_index_size])
       c%val = 0.0_real32
       do concurrent(s=1:size(indices), i=1:size(a%val,1))
          c%val(i, s) = a%val(i, indices(s))
       end do
    else
       call c%allocate(array_shape=[size(a%val,1), new_index_size])
       c%val = 0.0_real32
       do concurrent(s=1:size(indices), i=1:size(a%val,1))
          c%val(i, indices(s)) = a%val(i, s)
       end do
    end if
    c%indices = indices

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'index'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function reverse_index_array
!###############################################################################


!###############################################################################
  function pack_array(a, indices, dim) result(c)
    !! Pack an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, dimension(:), intent(in) :: indices
    integer, intent(in) :: dim
    type(array_type), pointer :: c

    integer :: i, s

    if(dim.eq.1)then
       c => a%create_result(array_shape=[size(indices), size(a%val,2)])
       do concurrent(s=1:size(a%val,2), i=1:size(indices))
          c%val(i, s) = a%val(indices(i), s)
       end do
    elseif(dim.eq.2)then
       c => a%create_result(array_shape=[size(a%val,1), size(indices)])
       do concurrent(s=1:size(indices), i=1:size(a%val,1))
          c%val(i, s) = a%val(i, indices(s))
       end do
    end if
    c%indices = indices
    allocate(c%adj_ja(2,1))
    c%adj_ja(:,1) = [ dim, size(a%val,dim) ]

    c%get_partial_left => get_partial_pack
    c%get_partial_right => get_partial_unpack
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'pack'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function pack_array
!-------------------------------------------------------------------------------
  function get_partial_pack(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => unpack(upstream_grad, this%indices, this%adj_ja(1,1), this%adj_ja(2,1))
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_pack
!###############################################################################


!###############################################################################
  function unpack_array(a, indices, dim, new_size) result(c)
    !! Unpack an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, dimension(:), intent(in) :: indices
    integer, intent(in) :: new_size, dim
    type(array_type), pointer :: c

    integer :: i, s


    if(dim.eq.1)then
       c => a%create_result(array_shape = [ new_size, size(a%val,2) ])
       c%val = 0.0_real32
       do concurrent(i=1:size(indices,1), s=1:size(a%val,2))
          c%val(indices(i),s) = a%val(i,s)
       end do
    elseif(dim.eq.2)then
       c => a%create_result(array_shape = [ size(a%val,1), new_size ])
       c%val = 0.0_real32
       do concurrent(i=1:size(a%val,1), s=1:new_size)
          c%val(i,indices(s)) = a%val(i,s)
       end do
    end if
    c%indices = indices
    allocate(c%adj_ja(2,1))
    c%adj_ja(:,1) = [ dim, new_size ]

    c%get_partial_left => get_partial_unpack
    c%get_partial_right => get_partial_pack
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'unpack'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function unpack_array
!-------------------------------------------------------------------------------
  function get_partial_unpack(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => pack(upstream_grad, this%indices, this%adj_ja(1,1))
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_unpack
!###############################################################################

end module diffstruc__operations_broadcast
