module diffstruc__operations_broadcast
  !! This module contains broadcast operations for the diffstruc library.
  use coreutils, only: real32
  use diffstruc__types, only: array_type
  implicit none


  private

  public :: concat, slice_left, slice_right, ltrim, rtrim, &
       operator(.index.), reverse_index, &
       pack, unpack, reshape

  ! Operation interfaces
  !-----------------------------------------------------------------------------
  interface concat
     module function concat_arrays(a, b, dim) result(c)
       class(array_type), intent(in), target :: a, b
       integer, intent(in), optional :: dim
       type(array_type), pointer :: c
     end function concat_arrays
  end interface

  interface slice_left
     module function slice_left_array(a, b, dim) result(c)
       class(array_type), intent(in), target :: a
       integer, intent(in) :: b
       integer, intent(in), optional :: dim
       type(array_type), pointer :: c
     end function slice_left_array
  end interface

  interface slice_right
     module function slice_right_array(a, b, dim) result(c)
       class(array_type), intent(in), target :: a
       integer, intent(in) :: b
       integer, intent(in), optional :: dim
       type(array_type), pointer :: c
     end function slice_right_array
  end interface

  interface ltrim
     module function ltrim_array(a, b, dim) result(c)
       class(array_type), intent(in), target :: a
       integer, intent(in) :: b
       integer, intent(in), optional :: dim
       type(array_type), pointer :: c
     end function ltrim_array
  end interface

  interface rtrim
     module function rtrim_array(a, b, dim) result(c)
       class(array_type), intent(in), target :: a
       integer, intent(in) :: b
       integer, intent(in), optional :: dim
       type(array_type), pointer :: c
     end function rtrim_array
  end interface

  interface operator(.index.)
     module function index_array(a, indices) result(c)
       class(array_type), intent(in), target :: a
       integer, dimension(:), intent(in) :: indices
       type(array_type), pointer :: c
     end function index_array
  end interface

  interface reverse_index
     module function reverse_index_array(a, indices, from, new_index_size) result(c)
       class(array_type), intent(in), target :: a
       integer, dimension(:), intent(in) :: indices
       logical, intent(in) :: from
       integer, intent(in) :: new_index_size
       type(array_type), pointer :: c
     end function reverse_index_array
  end interface

  interface pack
     module function pack_mask_array(a, dim, mask) result(c)
       class(array_type), intent(in), target :: a
       integer, intent(in) :: dim
       logical, dimension(:), intent(in), optional :: mask
       type(array_type), pointer :: c
     end function pack_mask_array

     module function pack_indices_array(a, indices, dim) result(c)
       class(array_type), intent(in), target :: a
       integer, dimension(:), intent(in) :: indices
       integer, intent(in) :: dim
       type(array_type), pointer :: c
     end function pack_indices_array
  end interface

  interface unpack
     module function unpack_mask_array(a, array_shape, dim, indices) result(c)
       class(array_type), intent(in), target :: a
       integer, dimension(:), intent(in) :: array_shape
       integer, intent(in), optional :: dim
       integer, dimension(:), intent(in), optional :: indices
       type(array_type), pointer :: c
     end function unpack_mask_array

     module function unpack_indices_array(a, indices, dim, new_size) result(c)
       class(array_type), intent(in), target :: a
       integer, dimension(:), intent(in) :: indices
       integer, intent(in) :: new_size, dim
       type(array_type), pointer :: c
     end function unpack_indices_array
  end interface

  interface reshape
     module function reshape_array(a, new_shape) result(c)
       class(array_type), intent(in), target :: a
       integer, dimension(:), intent(in) :: new_shape
       type(array_type), pointer :: c
     end function reshape_array
  end interface

end module diffstruc__operations_broadcast
