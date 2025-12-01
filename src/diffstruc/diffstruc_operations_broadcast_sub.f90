submodule(diffstruc__operations_broadcast) diffstruc__operations_broadcast_sub
  !! Submodule containing implementations of broadcast operations
  use coreutils, only: stop_program


contains

!###############################################################################
  module function concat_arrays(a, b, dim) result(c)
    !! Concatenate two autodiff arrays along the first dimension
    implicit none
    class(array_type), intent(in), target :: a, b
    integer, intent(in), optional :: dim
    type(array_type), pointer :: c

    integer :: i, s
    integer :: dim_

    if(present(dim)) then
       dim_ = dim
    else
       dim_ = 1
    end if

    ! concatenate 1D array by using shape to swap dimensions
    if(dim_.eq.1)then
       c => a%create_result(array_shape = &
            [size(a%val,1) + size(b%val,1), size(a%val,2)])
       c%val = 0._real32
       do concurrent(s=1:size(a%val,2))
          do concurrent(i=1:size(a%val,1))
             c%val(i, s) = a%val(i, s)
          end do
          do concurrent(i=1:size(b%val,1))
             c%val( size(a%val,1) + i, s) = b%val( i, s)
          end do
       end do
    else
       c => a%create_result(array_shape = &
            [size(a%val,1), size(a%val,2) + size(b%val,2)])
       c%val = 0._real32
       do concurrent(s=1:size(a%val,1))
          do concurrent(i=1:size(a%val,2))
             c%val(s, i) = a%val(s, i)
          end do
          do concurrent(i=1:size(b%val,2))
             c%val( s, size(a%val,2) + i) = b%val( s, i)
          end do
       end do
    end if
    c%indices = [ dim_ ]

    c%get_partial_left => get_partial_concat_left
    c%get_partial_right => get_partial_concat_right
    c%get_partial_left_val => get_partial_concat_left_val
    c%get_partial_right_val => get_partial_concat_right_val
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

    output = slice_left(upstream_grad, this%left_operand%shape(1), this%indices(1))

  end function get_partial_concat_left
!-------------------------------------------------------------------------------
  function get_partial_concat_right(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => slice_right(upstream_grad, this%right_operand%shape(1), this%indices(1))
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_concat_right
!-------------------------------------------------------------------------------
  pure subroutine get_partial_concat_left_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    if(this%indices(1).eq.1)then
       output = upstream_grad(1:size(this%left_operand%val,1), :)
    else
       output = upstream_grad(:, 1:size(this%left_operand%val,2))
    end if

  end subroutine get_partial_concat_left_val
!-------------------------------------------------------------------------------
  pure subroutine get_partial_concat_right_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    if(this%indices(1).eq.1)then
       output = upstream_grad( &
            size(this%left_operand%val,1)+1:size(upstream_grad,1), : &
       )
    else
       output = upstream_grad(:, &
            size(this%left_operand%val,2)+1:size(upstream_grad,2) &
       )
    end if

  end subroutine get_partial_concat_right_val
!###############################################################################


!###############################################################################
  module function slice_left_array(a, b, dim) result(c)
    !! Left trim an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: b
    integer, intent(in), optional :: dim
    type(array_type), pointer :: c

    integer :: s
    integer :: dim_

    if(present(dim)) then
       dim_ = dim
    else
       dim_ = 1
    end if

    ! left trim 1D array by using shape to swap dimensions
    if(dim_.eq.1)then
       c => a%create_result(array_shape = [b, size(a%val,2)])
       do concurrent(s=1:size(a%val,2))
          c%val( :, s) = a%val( 1:b, s)
       end do
    else
       c => a%create_result(array_shape = [size(a%val,1), b])
       do concurrent(s=1:size(a%val,1))
          c%val( s, : ) = a%val( s, 1:b)
       end do
    end if
    c%indices = [ dim_, b ]

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'slice_left'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function slice_left_array
!###############################################################################


!###############################################################################
  module function slice_right_array(a, b, dim) result(c)
    !! Right trim an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: b
    integer, intent(in), optional :: dim
    type(array_type), pointer :: c

    integer :: s
    integer :: dim_

    if(present(dim)) then
       dim_ = dim
    else
       dim_ = 1
    end if

    ! right trim 1D array by using shape to swap dimensions
    if(dim_.eq.1)then
       c => a%create_result(array_shape = [b, size(a%val,2)])
       do concurrent(s=1:size(a%val,2))
          c%val( :, s) = a%val( size(a%val,1)-b+1:size(a%val,1), s)
       end do
    else
       c => a%create_result(array_shape = [size(a%val,1), b])
       do concurrent(s=1:size(a%val,1))
          c%val( s, : ) = a%val( s, size(a%val,2)-b+1:size(a%val,2))
       end do
    end if
    c%indices = [ dim_, b ]

    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'slice_right'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function slice_right_array
!###############################################################################


!###############################################################################
  module function ltrim_array(a, b, dim) result(c)
    !! Left trim an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: b
    integer, intent(in), optional :: dim
    type(array_type), pointer :: c

    if(present(dim)) then
       c => slice_right_array(a, size(a%val, dim) - b, dim)
    else
       c => slice_right_array(a, size(a%val, 1) - b, 1)
    end if

  end function ltrim_array
!###############################################################################


!###############################################################################
  module function rtrim_array(a, b, dim) result(c)
    !! Right trim an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: b
    integer, intent(in), optional :: dim
    type(array_type), pointer :: c

    if(present(dim)) then
       c => slice_left_array(a, size(a%val, dim) - b, dim)
    else
       c => slice_left_array(a, size(a%val, 1) - b, 1)
    end if

  end function rtrim_array
!###############################################################################


!###############################################################################
  module function index_array(a, indices) result(c)
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
  module function reverse_index_array(a, indices, from, new_index_size) result(c)
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
  module function pack_mask_array(a, dim, mask) result(c)
    !! Pack an autodiff array using a logical mask
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    logical, dimension(:), intent(in), optional :: mask
    type(array_type), pointer :: c

    integer :: i, j, s, itmp1

    if(present(mask))then
       itmp1 = count(mask)
       if(dim.eq.1)then
          c => a%create_result(array_shape=[itmp1, size(a%val,2)])
          allocate(c%indices(itmp1))
          i = 0
          do concurrent(s=1:size(a%val,2))
             do j = 1, size(a%val,1)
                if(mask(j)) then
                   i = i + 1
                   c%val(i, s) = a%val(j, s)
                end if
             end do
             i = 0
          end do
          c%indices = pack([(j, j=1,size(mask))], mask)
       elseif(dim.eq.2)then
          c => a%create_result(array_shape=[a%shape, itmp1])
          allocate(c%indices(itmp1))
          i = 0
          do concurrent(s=1:size(a%val,1))
             do j = 1, size(a%val,2)
                if(mask(j)) then
                   i = i + 1
                   c%val(s, i) = a%val(s, j)
                end if
             end do
             i = 0
          end do
          c%indices = pack([(j, j=1,size(mask))], mask)
       else
          call stop_program("pack_mask: only 1 or 2 dimensions are supported")
       end if
    else
       if(dim.eq.1)then
          c => a%create_result(array_shape=[size(a%val,1), size(a%val,2)])
          do concurrent(s=1:size(a%val,2), i=1:size(a%val,1))
             c%val(i, s) = a%val(i, s)
          end do
       elseif(dim.eq.2)then
          c => a%create_result()
          do concurrent(s=1:size(a%val,1), i=1:size(a%val,2))
             c%val(s, i) = a%val(s, i)
          end do
       else
          call stop_program("pack: only 1 or 2 dimensions are supported")
       end if
       !c%indices = [(j, j=1,size(a%val,dim))]
    end if
    allocate(c%adj_ja(size(a%shape)+2,1))
    c%adj_ja(1,1) = dim
    c%adj_ja(2:,1) = [ a%shape, size(a%val,2) ]

    c%get_partial_left => get_partial_pack_mask
    c%get_partial_left_val => get_partial_pack_mask_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'pack_mask'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function pack_mask_array
!-------------------------------------------------------------------------------
  function get_partial_pack_mask(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    if(allocated(this%indices))then
       ptr => unpack(upstream_grad, array_shape=this%adj_ja(2:,1), &
            dim = this%adj_ja(1,1), &
            indices = this%indices &
       )
    else
       ptr => unpack(upstream_grad, array_shape=this%adj_ja(2:,1), &
            dim = this%adj_ja(1,1))
    end if
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_pack_mask
!-------------------------------------------------------------------------------
  pure subroutine get_partial_pack_mask_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, j, s, itmp1

    output = 0.0_real32
    if(allocated(this%indices))then
       itmp1 = size(this%indices)
       if(this%adj_ja(1,1).eq.1)then
          do concurrent(s=1:size(upstream_grad,2), j=1:itmp1)
             output(this%indices(j), s) = upstream_grad(j, s)
          end do
       elseif(this%adj_ja(1,1).eq.2)then
          do concurrent(s=1:itmp1, j=1:size(upstream_grad,1))
             output(j, this%indices(s)) = upstream_grad(j, s)
          end do
       end if
    end if
  end subroutine get_partial_pack_mask_val
!###############################################################################


!###############################################################################
  module function unpack_mask_array(a, array_shape, dim, indices) result(c)
    !! Unpack an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, dimension(:), intent(in) :: array_shape
    integer, intent(in), optional :: dim
    integer, dimension(:), intent(in), optional :: indices
    type(array_type), pointer :: c

    integer :: i, s, dim_, num_samples, num_elements

    if(present(dim)) then
       dim_ = dim
    else
       dim_ = 1
    end if
    num_samples = array_shape(size(array_shape))
    num_elements = product(array_shape(1:size(array_shape)-1))

    c => a%create_result(array_shape = array_shape)
    c%val = 0.0_real32
    if(dim_.eq.1)then
       if(present(indices))then
          do concurrent(i=1:size(indices,1), s=1:num_samples)
             c%val(indices(i),s) = a%val(i,s)
          end do
       else
          do concurrent(i=1:num_elements, s=1:num_samples)
             c%val(i,s) = a%val(i,s)
          end do
       end if
    elseif(dim_.eq.2)then
       if(present(indices))then
          do concurrent(i=1:num_elements, s=1:size(indices,1))
             c%val(i,indices(s)) = a%val(i,s)
          end do
       else
          do concurrent(i=1:num_elements, s=1:num_samples)
             c%val(i,s) = a%val(i,s)
          end do
       end if
    else
       call stop_program("unpack_mask: only 1 or 2 dimensions are supported")
    end if

    if(present(indices))then
       c%indices = indices
    end if
    allocate(c%adj_ja(1,1))
    c%adj_ja(1,1) = dim_

    c%get_partial_left => get_partial_unpack_mask
    c%get_partial_right => get_partial_pack_mask
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'unpack_mask'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function unpack_mask_array
!-------------------------------------------------------------------------------
  function get_partial_unpack_mask(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    logical, dimension(:), allocatable :: mask

    if(allocated(this%indices))then
       allocate(mask(size(upstream_grad%val,this%adj_ja(1,1))))
       mask = .false.
       mask(this%indices) = .true.
       ptr => pack(upstream_grad, this%adj_ja(1,1), mask)
    else
       ptr => pack(upstream_grad, this%adj_ja(1,1))
    end if

    call output%assign_and_deallocate_source(ptr)
  end function get_partial_unpack_mask
!###############################################################################



!###############################################################################
  module function pack_indices_array(a, indices, dim) result(c)
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

    c%get_partial_left => get_partial_pack_indices
    c%get_partial_right => get_partial_unpack_indices
    c%get_partial_left_val => get_partial_pack_indices_val
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'pack_indices'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function pack_indices_array
!-------------------------------------------------------------------------------
  function get_partial_pack_indices(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => unpack(upstream_grad, this%indices, this%adj_ja(1,1), this%adj_ja(2,1))
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_pack_indices
!-------------------------------------------------------------------------------
  pure subroutine get_partial_pack_indices_val(this, upstream_grad, output)
    implicit none
    class(array_type), intent(in) :: this
    real(real32), dimension(:,:), intent(in) :: upstream_grad
    real(real32), dimension(:,:), intent(out) :: output

    integer :: i, s
    integer :: dim, new_size

    dim = this%adj_ja(1,1)
    new_size = this%adj_ja(2,1)

    output = 0.0_real32
    if(dim.eq.1)then
       do concurrent(i=1:size(this%indices,1), s=1:size(upstream_grad,2))
          output(this%indices(i),s) = upstream_grad(i,s)
       end do
    elseif(dim.eq.2)then
       do concurrent(i=1:size(upstream_grad,1), s=1:new_size)
          output(i,this%indices(s)) = upstream_grad(i,s)
       end do
    end if

  end subroutine get_partial_pack_indices_val
!###############################################################################


!###############################################################################
  module function unpack_indices_array(a, indices, dim, new_size) result(c)
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

    c%get_partial_left => get_partial_unpack_indices
    c%get_partial_right => get_partial_pack_indices
    if(a%requires_grad) then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'unpack_indices'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function unpack_indices_array
!-------------------------------------------------------------------------------
  function get_partial_unpack_indices(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => pack(upstream_grad, this%indices, this%adj_ja(1,1))
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_unpack_indices
!###############################################################################

end submodule diffstruc__operations_broadcast_sub
