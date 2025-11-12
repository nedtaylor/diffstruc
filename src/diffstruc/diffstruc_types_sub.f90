submodule(diffstruc__types) diffstruc__types_submodule
  !! Submodule containing implementations for derived types
  use coreutils, only: stop_program, print_warning
  use diffstruc__global, only: max_recursion_depth, default_map_capacity



contains

!###############################################################################
  module subroutine allocate_array(this, array_shape, source)
    !! Allocate array
    implicit none

    ! Arguments
    class(array_type), intent(inout), target :: this
    !! Instance of the array type
    integer, dimension(:), intent(in), optional :: array_shape
    !! Shape of the array
    class(*), dimension(..), intent(in), optional :: source
    !! Source array

    if(allocated(this%val).or.this%allocated)then
       call stop_program("Trying to allocate already allocated array values")
       return
    end if
    if(present(array_shape))then
       allocate(this%val( &
            product(array_shape(1:size(array_shape)-1)),  &
            array_shape(size(array_shape)) &
       ))
       this%shape = array_shape(1:size(array_shape)-1)
    end if
    if(present(source))then
       select rank(source)
       rank(0)
          select type(source)
          type is (real(real32))
             if(.not.present(array_shape))then
                call stop_program('Source shape not provided')
                return
             end if
             this%val(:,:) = source
          type is (array_type)
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source%val)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             end if
             this = source
             !  this%val_ptr( &
             !       1:source%shape(1), &
             !       1:size(source%val, dim=2) &
             !  ) => this%val
          class default
             call stop_program('Incompatible source type for rank 0')
             return
          end select
       rank(2)
          select type(source)
          type is (real(real32))
             if(present(array_shape))then
                if(any(array_shape.ne.shape(source)))then
                   call stop_program('Source shape does not match array shape')
                   return
                end if
             end if
             this%val = source
             !  this%val_ptr( &
             !       1:size(source, dim=1), &
             !       1:size(source, dim=2) &
             !  ) => this%val
          class default
             call stop_program('Incompatible source type for rank 2')
             return
          end select
       rank default
          call stop_program('Unrecognised source rank')
          return
       end select
    end if
    if(.not.present(source).and..not.present(array_shape))then
       call stop_program('No shape or source provided')
       return
    end if
    this%rank = 1
    this%allocated = .true.
    !  this%val_ptr(1:size(this%val, dim=1), 1:size(this%val, dim=2)) => this%val
    if(.not.allocated(this%shape)) this%shape = [ size(this%val, dim=1) ]
    this%size = product(this%shape)


  end subroutine allocate_array
!###############################################################################


!###############################################################################
  module recursive subroutine deallocate_array(this, keep_shape)
    !! Deallocate array
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    !! Instance of the array type
    logical, intent(in), optional :: keep_shape
    !! Boolean whether to keep shape

    ! Local variables
    logical :: keep_shape_
    !! Boolean whether to keep shape

    keep_shape_ = .false.
    if(present(keep_shape)) keep_shape_ = keep_shape

    ! Deallocate all allocatable arrays
    if(allocated(this%val)) deallocate(this%val)
    if(allocated(this%indices)) deallocate(this%indices)
    if(allocated(this%adj_ja)) deallocate(this%adj_ja)
    if(allocated(this%mask)) deallocate(this%mask)
    if(allocated(this%direction)) deallocate(this%direction)
    if(.not.keep_shape_)then
       if(allocated(this%shape)) deallocate(this%shape)
    end if

    this%allocated = .false.
    this%size = 0
    ! write(*,*) "deallocated array loc: ", loc(this)

  end subroutine deallocate_array
!###############################################################################


!###############################################################################
  module recursive subroutine finalise_array(this)
    !! Finalise array - clean up memory safely
    implicit none
    type(array_type), intent(inout) :: this

    ! write(*,*) "finalising array loc: ", loc(this), this%is_temporary, this%operation

    ! First deallocate owned pointers (operands and gradient) to prevent leaks
    if(.not.this%is_temporary)then
       if(associated(this%left_operand))then
          if(this%owns_left_operand)then
             !call this%left_operand%deallocate()
             deallocate(this%left_operand)
          end if
          nullify(this%left_operand)
       end if

       if(associated(this%right_operand))then
          if(this%owns_right_operand)then
             !call this%right_operand%deallocate()
             deallocate(this%right_operand)
          end if
          nullify(this%right_operand)
       end if

       if(associated(this%grad))then
          if(this%owns_gradient)then
             !call this%grad%deallocate()
             deallocate(this%grad)
          end if
          nullify(this%grad)
       end if
    end if

    ! Deallocate all allocatable arrays
    call this%deallocate()

    ! Reset ownership flags and nullify procedure pointers
    this%owns_gradient = .false.
    this%owns_left_operand = .false.
    this%owns_right_operand = .false.
    nullify(this%get_partial_left)
    nullify(this%get_partial_right)
    this%is_temporary = .true.
    ! write(*,*) "finalised array loc: ", loc(this)

  end subroutine finalise_array
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  pure module function flatten_array(this) result(output)
    !! Flatten the array
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    !! Instance of the array type
    real(real32), dimension(this%size) :: output
    !! Flattened array

    output = reshape(this%val, [this%size])
  end function flatten_array
!###############################################################################


!###############################################################################
  module recursive subroutine assign_array(this, input)
    !! Assign the array
    implicit none

    ! Arguments
    class(array_type), intent(out), target :: this
    !! Instance of the array type
    type(array_type), intent(in) :: input
    !! Input array

    this%id = input%id
    this%rank = input%rank
    this%size = input%size
    this%is_sample_dependent = input%is_sample_dependent
    this%is_forward = input%is_forward
    this%is_scalar = input%is_scalar
    this%allocated = input%allocated
    if(allocated(input%shape)) this%shape = input%shape
    if(allocated(input%val)) this%val = input%val
    this%requires_grad = input%requires_grad
    if(associated(input%grad)) this%grad => input%grad
    if(associated(input%left_operand)) this%left_operand => input%left_operand
    if(associated(input%right_operand)) this%right_operand => input%right_operand
    this%operation = input%operation

    ! Transfer ownership flags - this is critical to prevent memory leaks
    this%owns_gradient = input%owns_gradient
    this%owns_left_operand = input%owns_left_operand
    this%owns_right_operand = input%owns_right_operand

    if(allocated(input%indices)) this%indices = input%indices
    if(allocated(input%adj_ja)) this%adj_ja = input%adj_ja
    if(allocated(input%mask)) this%mask = input%mask

    if(associated(input%get_partial_left)) &
         this%get_partial_left => input%get_partial_left
    if(associated(input%get_partial_right)) &
         this%get_partial_right => input%get_partial_right

  end subroutine assign_array
!-------------------------------------------------------------------------------
  module subroutine assign_shallow(this, source)
    !! Assign the array
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    !! Instance of the array type
    class(array_type), intent(in), target :: source
    !! source array

    this%id = source%id
    this%rank = source%rank
    this%size = source%size
    this%is_sample_dependent = source%is_sample_dependent
    this%is_forward = source%is_forward
    this%is_scalar = source%is_scalar
    this%allocated = source%allocated
    if(allocated(source%shape)) this%shape = source%shape
    if(allocated(source%val)) this%val = source%val
    this%requires_grad = source%requires_grad
    if(associated(source%grad)) this%grad => source%grad
    if(associated(source%left_operand)) this%left_operand => source%left_operand
    if(associated(source%right_operand)) this%right_operand => source%right_operand
    this%operation = source%operation
    this%owns_gradient = .false.  ! Dont copy gradient ownership
    if(allocated(source%indices)) this%indices = source%indices
    if(allocated(source%adj_ja)) this%adj_ja = source%adj_ja
    if(allocated(source%mask)) this%mask = source%mask
    this%owns_left_operand = source%owns_left_operand
    this%owns_right_operand = source%owns_right_operand

    if(associated(source%get_partial_left)) &
         this%get_partial_left => source%get_partial_left
    if(associated(source%get_partial_right)) &
         this%get_partial_right => source%get_partial_right

  end subroutine assign_shallow
!-------------------------------------------------------------------------------
  module subroutine assign_and_deallocate_source(this, source, owns_left_operand, &
       owns_right_operand)
    !! Assign and deallocate the source array
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    !! Instance of the array type
    type(array_type), intent(inout), pointer :: source
    !! Source array
    logical, intent(in), optional :: owns_left_operand, owns_right_operand

    if(present(owns_left_operand)) source%owns_left_operand = owns_left_operand
    if(present(owns_right_operand)) source%owns_right_operand = owns_right_operand
    this = source
    deallocate(source)
    ! nullify(source)

  end subroutine assign_and_deallocate_source
!###############################################################################


!###############################################################################
  module function create_result_array(this, array_shape) result(result_ptr)
    !! Helper function to safely create result arrays with proper initialization
    implicit none
    class(array_type), intent(in) :: this
    integer, dimension(:), intent(in), optional :: array_shape
    type(array_type), pointer :: result_ptr

    allocate(result_ptr)

    if(present(array_shape))then
       call result_ptr%allocate(array_shape=array_shape)
    else
       if(allocated(this%shape))then
          call result_ptr%allocate(array_shape=[this%shape, &
               size(this%val,2)])
       else
          call result_ptr%allocate(array_shape=shape(this%val))
       end if
    end if

    ! Initialise autodiff fields
    result_ptr%requires_grad = .false.
    result_ptr%is_scalar = this%is_scalar
    result_ptr%is_sample_dependent = this%is_sample_dependent
    result_ptr%is_forward = this%is_forward
    result_ptr%operation = 'none'
    result_ptr%owns_gradient = .false.
    result_ptr%owns_left_operand = .false.
    result_ptr%owns_right_operand = .false.
    result_ptr%left_operand => null()
    result_ptr%right_operand => null()
    result_ptr%get_partial_left => null()
    result_ptr%get_partial_right => null()
    result_ptr%is_temporary = .true.
  end function create_result_array
!###############################################################################


!###############################################################################
  pure module subroutine set_array(this, input)
    !! Set the array
    implicit none

    ! Arguments
    class(array_type), intent(inout) :: this
    !! Instance of the array type
    real(real32), dimension(..), intent(in) :: input
    !! Input array

    if( any(shape(input).ne.[this%shape, size(this%val,2)]) )then
       return
    end if
    select rank(input)
    rank(1)
       this%val(:,1) = input
    rank(2)
       this%val(:,:) = input
    rank(3)
       this%val(:,:) = reshape(input, shape(this%val))
    rank(4)
       this%val(:,:) = reshape(input, shape(this%val))
    rank(5)
       this%val(:,:) = reshape(input, shape(this%val))
    end select
  end subroutine set_array
!###############################################################################


!###############################################################################
  module subroutine extract_array(this, output)
    !! Extract the array
    implicit none

    ! Arguments
    class(array_type), intent(in) :: this
    !! Instance of the array type
    real(real32), dimension(..), allocatable, intent(out) :: output
    !! Output array

    ! Local variables
    character(len=10) :: rank_str
    !! String for rank


    select rank(output)
    rank(1)
       output = reshape(this%val, [ product(this%shape) * size(this%val,2) ])
    rank(2)
       output = this%val
    rank default
       if(size(this%shape,1) + 1 .ne.rank(output))then
          write(rank_str,'(I0)') rank(output)
          call print_warning( &
               "Output data rank mismatch, expected rank "//trim(adjustl(rank_str)) &
          )
          return
       end if
       select rank(output)
       rank(3)
          output = reshape( &
               this%val, &
               [ &
                    this%shape(1), &
                    this%shape(2), &
                    size(this%val,2) &
               ] &
          )
       rank(4)
          output = reshape( &
               this%val, &
               [ &
                    this%shape(1), &
                    this%shape(2), &
                    this%shape(3), &
                    size(this%val,2) &
               ] &
          )
       rank(5)
          output = reshape( &
               this%val, &
               [ &
                    this%shape(1), &
                    this%shape(2), &
                    this%shape(3), &
                    this%shape(4), &
                    size(this%val,2) &
               ] &
          )
       end select
    end select

  end subroutine extract_array
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function grad_forward(this, variable) result(output)
    !! Perform forward-mode automatic differentiation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: variable
    type(array_type), pointer :: output

    integer :: depth

    depth = 0
    output => forward_over_reverse(this, variable, depth)
    output%is_forward = .true.
    this%requires_grad = .true.

  end function grad_forward
!###############################################################################


!###############################################################################
  module subroutine grad_reverse(this, record_graph, reset_graph)
    !! Perform backward pass starting from this array
    implicit none
    class(array_type), intent(inout) :: this
    logical, intent(in), optional :: record_graph
    logical, intent(in), optional :: reset_graph

    logical :: record_graph_

    record_graph_ = .true.
    if(present(record_graph)) record_graph_ = record_graph
    if(present(reset_graph))then
       if(reset_graph) call this%reset_graph()
    end if
    call zero_all_fixed_pointer_grads(this)

    ! Initialise gradient if not allocated
    if(.not. associated(this%grad))then
       allocate(this%grad)
       ! Safely initialise gradient without copying computation graph
       call this%grad%allocate(array_shape=[this%shape, size(this%val,2)])
       this%grad%is_sample_dependent = this%is_sample_dependent
       this%grad%requires_grad = record_graph_
       this%grad%operation = 'none'
       this%grad%left_operand => null()
       this%grad%right_operand => null()
       this%grad%get_partial_left => null()
       this%grad%get_partial_right => null()
       this%grad%grad => null()
       this%grad%owns_gradient = .false.
       this%owns_gradient = .true.
       if(allocated(this%indices)) this%grad%indices = this%indices
       call this%grad%zero_grad()
       ! Set gradient to ones for starting node
       this%grad%val = 1.0_real32
       this%grad%is_temporary = this%is_temporary
    end if

    ! Recursively compute gradients
    if(record_graph_)then
       call reverse_mode_ptr(this, this%grad, 0)
    else
       call reverse_mode(this, this%grad, 0)
    end if
  end subroutine grad_reverse
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  recursive function forward_over_reverse(this, variable, depth) result(output)
    implicit none
    type(array_type), intent(inout) :: this
    type(array_type), intent(in) :: variable
    integer, intent(inout) :: depth
    type(array_type), pointer :: output

    integer :: s
    logical :: is_right_a_variable, is_left_a_variable
    type(array_type), pointer :: left_deriv_tmp, right_deriv_tmp
    type(array_type), pointer :: left_deriv, right_deriv
    logical :: is_forward_local

    depth = depth + 1
    if(depth.gt.max_recursion_depth)then
       write(0,*) "MAX RECURSION DEPTH REACHED", depth
       return
    end if
    is_forward_local = this%is_forward
    this%is_forward = .true.
    ! write(*,*) "Performing forward-over-reverse operation for: ", trim(this%operation)
    if(loc(this).eq.loc(variable))then
       ! call output%allocate(array_shape=[this%shape, size(this%val,2)])
       allocate(output)
       output = this
       if(allocated(this%direction))then
          do s = 1, size(output%val,2)
             output%val(:,s) = this%direction
          end do
       else
          output%val(:,:) = 1._real32
       end if
       if(allocated(output%direction)) deallocate(output%direction)

    elseif(associated(this%left_operand).or.associated(this%right_operand))then
       ! if(associated(this%grad))then
       !    output = this%grad
       ! else
       is_left_a_variable = .false.
       if(associated(this%left_operand))then
          if(associated(this%get_partial_left))then
             is_left_a_variable = .true.
             !if(associated(this%left_operand%grad))then
             !   left_deriv => this%left_operand%grad
             !else
             left_deriv_tmp => &
                  forward_over_reverse(this%left_operand, variable, depth)
             ! call left_deriv_tmp%set_requires_grad(.false.)
             if(trim(this%operation).eq.'add')then
                left_deriv => left_deriv_tmp
             else
                allocate(left_deriv)
                if( associated(this%get_partial_right) .and. &
                     .not.associated(this%right_operand) &
                )then
                   left_deriv = this%get_partial_right(left_deriv_tmp)
                else
                   left_deriv = this%get_partial_left(left_deriv_tmp)
                end if
             end if
             !write(*,*) "left owns l", left_deriv_tmp%owns_left_operand
             !write(*,*) "left owns r", left_deriv_tmp%owns_right_operand
             !  left_deriv%owns_left_operand = .true.
             !  left_deriv%owns_right_operand = .true.
             ! end if
          end if
       end if

       is_right_a_variable = .false.
       if(associated(this%right_operand))then
          if(associated(this%get_partial_right))then
             is_right_a_variable = .true.
             !if(associated(this%right_operand%grad))then
             !   right_deriv => this%right_operand%grad
             !else
             right_deriv_tmp => &
                  forward_over_reverse(this%right_operand, variable, depth)
             ! call right_deriv_tmp%set_requires_grad(.false.)
             if(trim(this%operation).eq.'add')then
                right_deriv => right_deriv_tmp
             else
                allocate(right_deriv)
                right_deriv = this%get_partial_right(right_deriv_tmp)
             end if
             !  right_deriv%owns_left_operand = .true.
             !  right_deriv%owns_right_operand = .true.
             !end if
          end if
       end if

       if(is_left_a_variable.and.is_right_a_variable)then
          output => left_deriv + right_deriv
       elseif(is_left_a_variable)then
          output => left_deriv
       elseif(is_right_a_variable)then
          output => right_deriv
       else
          call stop_program("Neither operand is a variable in forward-over-reverse")
       end if

       ! end if
    else
       allocate(output)
       output = this
       if(allocated(output%direction)) deallocate(output%direction)
       ! call output%allocate(array_shape=[this%shape, size(this%val,2)])
       output%val(:,:) = 0._real32
    end if
    this%is_forward = is_forward_local
    output%is_forward = .true.
    output%is_temporary = .true.
    ! write(*,*) "done operation: ", trim(this%operation)

  end function forward_over_reverse
!###############################################################################


!###############################################################################
  recursive subroutine reverse_mode_ptr(array, upstream_grad, depth)
    !! Backward operation for arrays
    implicit none
    class(array_type), intent(inout) :: array
    type(array_type), pointer, intent(in) :: upstream_grad
    integer, intent(in) :: depth

    type(array_type), pointer :: left_partial, right_partial

    ! write(*,'("Performing backward operation for: ",A,T60,"id: ",I0)') &
    !      trim(array%operation), array%id
    if(depth.gt.max_recursion_depth)then
       write(0,*) "MAX RECURSION DEPTH REACHED IN REVERSE MODE", depth
       return
    end if
    array%is_forward = .false.
    if(associated(array%left_operand))then
       if(array%left_operand%requires_grad)then
          allocate(left_partial)
          left_partial = array%get_partial_left(upstream_grad)
          left_partial%is_temporary = .true.
          call accumulate_gradient_ptr(array%left_operand, left_partial, depth)
       end if
    end if
    if(associated(array%right_operand))then
       if(array%right_operand%requires_grad)then
          allocate(right_partial)
          right_partial = array%get_partial_right(upstream_grad)
          right_partial%is_temporary = .true.
          call accumulate_gradient_ptr(array%right_operand, right_partial, depth)
       end if
    end if
    ! write(*,*) "done operation: ", trim(array%operation)
  end subroutine reverse_mode_ptr
!###############################################################################


!###############################################################################
  recursive subroutine accumulate_gradient_ptr(array, grad, depth)
    !! Accumulate gradient for array with safe memory management
    implicit none
    type(array_type), intent(inout) :: array
    type(array_type), intent(in), pointer :: grad
    integer, intent(in) :: depth

    integer :: s
    logical :: is_directional
    type(array_type), pointer :: directional_grad

    is_directional = .false.
    if(allocated(array%direction))then
       if(size(array%direction).gt.0) is_directional = .true.
    end if

    if(is_directional)then
       allocate(directional_grad)
       directional_grad = grad
       do s = 1, size(grad%val, 2)
          directional_grad%val(:, s) = grad%val(:, s) * array%direction
       end do
    else
       directional_grad => grad
    end if

    if(.not. associated(array%grad))then
       if(array%is_sample_dependent)then
          array%grad => directional_grad
       else
          ! ! mean reduction
          ! array%grad => array%grad + mean( directional_grad, dim = 2 )
          ! sum reduction
          array%grad => sum( directional_grad, dim = 2 )
       end if
       array%grad%is_scalar = array%is_scalar
       array%grad%is_sample_dependent = array%is_sample_dependent
       array%grad%requires_grad = .not. array%is_scalar
       array%grad%grad => null()
       array%grad%owns_gradient = .false.
       array%owns_gradient = .true.
       array%grad%is_temporary = array%is_temporary
    else

       array%grad%is_temporary = .true.
       if(array%is_sample_dependent)then
          array%grad => array%grad + directional_grad
       else
          ! ! mean reduction
          ! array%grad => array%grad + mean( directional_grad, dim = 2 )
          ! sum reduction
          array%grad => array%grad + sum( directional_grad%val, dim = 2 )
       end if
       array%grad%is_temporary = array%is_temporary

    end if

    if(associated(array%left_operand).or.associated(array%right_operand))then
       call reverse_mode_ptr(array, directional_grad, depth+1)
    end if
  end subroutine accumulate_gradient_ptr
!###############################################################################


!###############################################################################
  recursive subroutine reverse_mode(array, upstream_grad, depth)
    !! Backward operation for arrays
    implicit none
    class(array_type), intent(inout) :: array
    type(array_type), intent(in) :: upstream_grad
    integer, intent(in) :: depth

    type(array_type) :: left_partial, right_partial

    ! write(*,'("Performing backward operation for: ",A,T60,"id: ",I0)') &
    !      trim(array%operation), array%id
    if(depth.gt.max_recursion_depth)then
       write(0,*) "MAX RECURSION DEPTH REACHED IN REVERSE MODE", depth
       return
    end if
    array%is_forward = .false.
    if(associated(array%left_operand))then
       if(array%left_operand%requires_grad)then
          left_partial = array%get_partial_left(upstream_grad)
          left_partial%is_temporary = .false.
          call accumulate_gradient(array%left_operand, left_partial, depth)
       end if
    end if
    if(associated(array%right_operand))then
       if(array%right_operand%requires_grad)then
          right_partial = array%get_partial_right(upstream_grad)
          right_partial%is_temporary = .false.
          call accumulate_gradient(array%right_operand, right_partial, depth)
       end if
    end if
    ! write(*,*) "done operation: ", trim(array%operation)
  end subroutine reverse_mode
!###############################################################################


!###############################################################################
  recursive subroutine accumulate_gradient(array, grad, depth)
    !! Accumulate gradient for array with safe memory management
    implicit none
    type(array_type), intent(inout) :: array
    type(array_type), intent(inout) :: grad
    integer, intent(in) :: depth

    integer :: s

    ! Apply direction if specified (in-place to avoid copy)
    if(allocated(array%direction))then
       if(size(array%direction).gt.0)then
          do s = 1, size(grad%val, 2)
             grad%val(:, s) = grad%val(:, s) * array%direction
          end do
       end if
    end if

    if(.not. associated(array%grad))then
       ! First gradient accumulation - allocate and set
       allocate(array%grad)
       if(array%is_sample_dependent)then
          call array%grad%allocate(array_shape=[grad%shape,size(grad%val,2)])
          array%grad%val = grad%val
       else
          call array%grad%allocate(array_shape=[grad%shape,1])
          array%grad%val(:,1) = sum(grad%val, dim = 2)
       end if
       array%grad%is_scalar = array%is_scalar
       array%grad%is_sample_dependent = array%is_sample_dependent
       array%grad%requires_grad = .false.
       array%grad%owns_gradient = .false.
       array%owns_gradient = .true.
       array%grad%is_temporary = array%is_temporary
    else
       ! Accumulate to existing gradient (in-place)
       if(array%is_sample_dependent)then
          array%grad%val = array%grad%val + grad%val
       else
          ! rtmp1 = real(size(grad%val,2), real32)
          ! ! mean reduction
          ! do concurrent(s = 1:size(grad%val,1))
          !    array%grad%val(s,1) = array%grad%val(s,1) + sum(grad%val(s,:)) / rtmp1
          ! end do
          ! sum reduction
          array%grad%val(:,1) = array%grad%val(:,1) + sum(grad%val, dim = 2)
       end if
    end if

    if(associated(array%left_operand).or.associated(array%right_operand))then
       call reverse_mode(array, grad, depth+1)
    end if
    if(array%grad%is_temporary)then
       call array%grad%nullify_graph()
       call array%grad%deallocate()
       deallocate(array%grad)
       array%owns_gradient = .false.
    end if
  end subroutine accumulate_gradient
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module subroutine zero_grad(this)
    !! Zero the gradients of this array
    implicit none
    class(array_type), intent(inout) :: this

    if(associated(this%grad))then
       if(allocated(this%grad%val)) this%grad%val = 0.0_real32
    end if
  end subroutine zero_grad
!###############################################################################


!###############################################################################
  module recursive subroutine zero_all_grads(this)
    !! Zero the gradients of this array
    implicit none
    class(array_type), intent(inout) :: this

    if(associated(this%left_operand))then
       call this%left_operand%zero_all_grads()
    end if
    if(associated(this%right_operand))then
       call this%right_operand%zero_all_grads()
    end if
    if(associated(this%grad))then
       if(allocated(this%grad%val)) this%grad%val = 0.0_real32
    end if
  end subroutine zero_all_grads
!###############################################################################


!###############################################################################
  recursive subroutine zero_all_fixed_pointer_grads(this)
    !! Zero the gradients of this array
    implicit none
    type(array_type), intent(inout) :: this

    if(associated(this%left_operand))then
       call zero_all_fixed_pointer_grads(this%left_operand)
    end if
    if(associated(this%right_operand))then
       call zero_all_fixed_pointer_grads(this%right_operand)
    end if
    if(this%fix_pointer.and.associated(this%grad))then
       if(allocated(this%grad%val)) this%grad%val = 0.0_real32
    end if
  end subroutine zero_all_fixed_pointer_grads
!###############################################################################


!###############################################################################
  module recursive subroutine reset_graph(this)
    !! Reset the gradient graph of this array
    implicit none
    class(array_type), intent(inout) :: this

    if(associated(this%left_operand))then
       call this%left_operand%reset_graph()
    end if

    if(associated(this%right_operand))then
       call this%right_operand%reset_graph()
    end if

    call this%zero_grad()
    if(this%owns_gradient.and.associated(this%grad))then
       this%grad => null()
    end if

    ! Reset ownership flags
    this%owns_gradient = .false.

  end subroutine reset_graph
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
! Map helper functions for graph traversal
!###############################################################################
  subroutine double_map_add(src_map, dst_map, src_ptr, dst_ptr)
    !! Add pointer pair to double-array map (grow if needed)
    implicit none
    type(array_ptr), allocatable :: src_map(:), dst_map(:)
    type(array_type), pointer, intent(in) :: src_ptr, dst_ptr
    integer :: n, i, newcap
    if(.not. allocated(src_map)) allocate(src_map(default_map_capacity))
    if(.not. allocated(dst_map)) allocate(dst_map(default_map_capacity))
    n = size(src_map)
    ! find first null slot
    do i = 1, n
       if(.not. associated(src_map(i)%p))then
          src_map(i)%p => src_ptr
          dst_map(i)%p => dst_ptr
          return
       end if
    end do
    ! no slot -> grow (double capacity)
    src_map = [ src_map, array_ptr() ]
    dst_map = [ dst_map, array_ptr() ]
    ! store at next free
    src_map(n+1)%p => src_ptr
    dst_map(n+1)%p => dst_ptr
  end subroutine double_map_add
!-------------------------------------------------------------------------------
  subroutine single_map_add(map, ptr)
    !! Add pointer to single-array map (grow if needed)
    implicit none
    type(array_ptr), allocatable :: map(:)
    type(array_type), pointer, intent(in) :: ptr
    integer :: n, i

    if(.not. allocated(map))then
       allocate(map(default_map_capacity))
    end if
    n = size(map)
    ! find first null slot
    do i = 1, n
       if(.not. associated(map(i)%p))then
          map(i)%p => ptr
          return
       end if
    end do
    ! no slot -> grow
    map = [ map, array_ptr() ]
    map(n+1)%p => ptr
  end subroutine single_map_add
!-------------------------------------------------------------------------------
  function map_find(map, target) result(idx)
    !! Check if target pointer exists in single-array map
    implicit none
    type(array_ptr), allocatable :: map(:)
    type(array_type), pointer, intent(in) :: target
    integer :: idx, n, i

    idx = 0
    if(.not. allocated(map)) return
    n = size(map)
    do i = 1, n
       if(associated(map(i)%p, target))then
          idx = i
          return
       end if
    end do
  end function map_find
!-------------------------------------------------------------------------------
  subroutine map_free(map)
    !! Free array map (nullify pointers and deallocate array)
    implicit none
    type(array_ptr), allocatable :: map(:)
    integer :: i

    if(allocated(map))then
       do i = 1, size(map)
          if(associated(map(i)%p))then
             nullify(map(i)%p)
          end if
       end do
       deallocate(map)
    end if
  end subroutine map_free
!###############################################################################


!###############################################################################
  recursive subroutine nullify_graph_recursive(this, visited_map, dealloc_list, &
       ignore_ownership &
  )
    !! Recursive helper that tracks visited nodes to avoid infinite loops
    !! Instead of deallocating immediately, collect nodes to deallocate in a second pass
    implicit none
    class(array_type), intent(inout), target :: this
    type(array_ptr), allocatable, intent(inout) :: visited_map(:), dealloc_list(:)
    logical, intent(in) :: ignore_ownership
    integer :: idx

    ! Check if we've already visited this node
    idx = map_find(visited_map, this)
    if(idx .ne. 0) return  ! Already processed, avoid infinite loop

    ! Mark this node as visited by adding to map BEFORE recursing
    call single_map_add(visited_map, this)

    ! Now process children recursively first
    ! ... this node is not reprocessed due to visited check
    if(ignore_ownership)then
       ! Ignore ownership flags during traversal
       if(associated(this%left_operand))then
          call nullify_graph_recursive( &
               this%left_operand, visited_map, dealloc_list, ignore_ownership &
          )
       end if
       if(associated(this%right_operand))then
          call nullify_graph_recursive( &
               this%right_operand, visited_map, dealloc_list, ignore_ownership &
          )
       end if
       if(associated(this%grad))then
          call nullify_graph_recursive( &
               this%grad, visited_map, dealloc_list, ignore_ownership &
          )
       end if
    else
       if(associated(this%left_operand).and.this%owns_left_operand)then
          call nullify_graph_recursive( &
               this%left_operand, visited_map, dealloc_list, ignore_ownership &
          )
       end if
       if(associated(this%right_operand).and.this%owns_right_operand)then
          call nullify_graph_recursive( &
               this%right_operand, visited_map, dealloc_list, ignore_ownership &
          )
       end if
       if(associated(this%grad).and.this%owns_gradient)then
          call nullify_graph_recursive( &
               this%grad, visited_map, dealloc_list, ignore_ownership &
          )
       end if
    end if

    ! After recursion, collect nodes that need deallocation
    ! Only add to dealloc list if we own it and it's not a fixed pointer
    ! Check if already in list to avoid double-free
    if(associated(this%left_operand))then
       if(.not.this%left_operand%fix_pointer .and. this%owns_left_operand)then
          idx = map_find(dealloc_list, this%left_operand)
          if(idx .eq. 0)then  ! Not already listed
             call single_map_add(dealloc_list, this%left_operand)
          end if
       end if
       nullify(this%left_operand)
    end if

    if(associated(this%right_operand))then
       if(.not.this%right_operand%fix_pointer .and. this%owns_right_operand)then
          idx = map_find(dealloc_list, this%right_operand)
          if(idx .eq. 0)then  ! Not already listed
             call single_map_add(dealloc_list, this%right_operand)
          end if
       end if
       nullify(this%right_operand)
    end if

    if(associated(this%grad))then
       if(.not.this%grad%fix_pointer .and. this%owns_gradient)then
          idx = map_find(dealloc_list, this%grad)
          if(idx .eq. 0)then  ! Not already listed
             call single_map_add(dealloc_list, this%grad)
          end if
       end if
       nullify(this%grad)
    end if

    ! Reset ownership and procedure pointers
    this%owns_left_operand = .false.
    this%owns_right_operand = .false.
    this%owns_gradient = .false.
    nullify(this%get_partial_left)
    nullify(this%get_partial_right)

  end subroutine nullify_graph_recursive
!###############################################################################


!###############################################################################
  module subroutine nullify_graph(this, ignore_ownership)
    !! Nullify graph by tracking visited nodes to avoid infinite recursion
    implicit none
    class(array_type), intent(inout), target :: this
    logical, intent(in), optional :: ignore_ownership
    type(array_ptr), allocatable :: visited_map(:), dealloc_list(:)
    type(array_type), pointer :: node_to_dealloc
    integer :: i, n
    logical :: ignore_ownership_

    ignore_ownership_ = .not.this%is_forward
    if(present(ignore_ownership)) ignore_ownership_ = ignore_ownership

    ! Initialise and run the recursive cleanup with tracking
    ! This will traverse the graph, nullify pointers, and collect nodes to deallocate
    call nullify_graph_recursive(this, visited_map, dealloc_list, ignore_ownership_)

    ! Now deallocate all collected nodes in a second pass
    ! This avoids issues with deallocating while traversing
    if(allocated(dealloc_list))then
       n = size(dealloc_list)
       do i = 1, n
          if(associated(dealloc_list(i)%p))then
             node_to_dealloc => dealloc_list(i)%p
             if(node_to_dealloc%allocated)then
                call node_to_dealloc%deallocate()
             end if
             node_to_dealloc%is_temporary = .true.
             deallocate(node_to_dealloc)
             nullify(dealloc_list(i)%p)
          end if
       end do
       deallocate(dealloc_list)
    end if

    ! Clean up the map
    call map_free(visited_map)

  end subroutine nullify_graph
!###############################################################################


!###############################################################################
  recursive function duplicate_pointer(input_ptr, src_map, dst_map, owns_self) &
       result(output_ptr)
    implicit none
    type(array_type), target :: input_ptr
    type(array_ptr), allocatable :: src_map(:), dst_map(:)
    type(array_type), pointer :: output_ptr
    logical, intent(out):: owns_self
    integer :: idx
    logical :: tmp_logical

    owns_self = .false.
    idx = map_find(src_map, input_ptr)
    if(idx .ne. 0)then
       output_ptr => dst_map(idx)%p
       return
    elseif(input_ptr%fix_pointer)then
       ! If pointer is fixed, do not duplicate; just return original
       output_ptr => input_ptr
       return
    end if

    ! Create duplicate node (caller may want ownership policies)
    allocate(output_ptr)
    call output_ptr%assign_shallow(input_ptr)
    ! Mark that output_ptr is a duplicate (so callers can deallocate later)
    !output_ptr%owns_self = .true.     ! <-- add this flag to array_type if helpful
    owns_self = .true.

    ! Add to map BEFORE recursing to handle cycles / shared nodes
    call double_map_add(src_map, dst_map, input_ptr, output_ptr)

    ! Now recursively duplicate children (use pointer assignment to map results)
    if(associated(input_ptr%left_operand))then
       output_ptr%left_operand => duplicate_pointer( &
            input_ptr%left_operand, src_map, dst_map, tmp_logical &
       )
       output_ptr%owns_left_operand = tmp_logical
    end if
    if(associated(input_ptr%right_operand))then
       output_ptr%right_operand => duplicate_pointer( &
            input_ptr%right_operand, src_map, dst_map, tmp_logical &
       )
       output_ptr%owns_right_operand = tmp_logical
    end if
    if(associated(input_ptr%grad))then
       output_ptr%grad => duplicate_pointer(&
            input_ptr%grad, src_map, dst_map, tmp_logical &
       )
       output_ptr%owns_gradient = .true.
    end if

  end function duplicate_pointer
!###############################################################################


!###############################################################################
  module function duplicate_graph(this) result(output_ptr)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_ptr), allocatable :: src_map(:), dst_map(:)
    type(array_type), pointer :: output_ptr
    logical :: tmp_logical

    output_ptr => duplicate_pointer(this, src_map, dst_map, tmp_logical)

    ! Clean up the maps
    call map_free(src_map)
    call map_free(dst_map)

  end function duplicate_graph
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module recursive function get_ptr_from_id(this, id) result(ptr)
    use iso_c_binding
    implicit none
    class(array_type), intent(in), target :: this
    integer, intent(in) :: id
    type(array_type), pointer :: ptr

    ptr => null()
    if(this%id .eq. id)then
       ptr => this
       return
    end if
    if(associated(this%left_operand))then
       ptr => this%left_operand%get_ptr_from_id(id)
       if(associated(ptr)) return
    end if
    if(associated(this%right_operand))then
       ptr => this%right_operand%get_ptr_from_id(id)
       if(associated(ptr)) return
    end if
  end function get_ptr_from_id
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module subroutine detach(this)
    !! Detach this array from the computation graph
    implicit none
    class(array_type), intent(inout) :: this

    this%requires_grad = .false.
    this%operation = 'none'
    this%left_operand => null()
    this%right_operand => null()
  end subroutine detach
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module subroutine set_requires_grad(this, requires_grad)
    !! Set the requires_grad flag
    implicit none
    class(array_type), intent(inout) :: this
    logical, intent(in) :: requires_grad

    this%requires_grad = requires_grad
  end subroutine set_requires_grad
!###############################################################################


!###############################################################################
  module subroutine set_direction(this, direction)
    !! Set the direction for the array (for higher-order derivatives)
    implicit none
    class(array_type), intent(inout) :: this
    real(real32), dimension(:), intent(in) :: direction

    if(allocated(this%direction)) deallocate(this%direction)
    if(size(this%val,1).ne.size(direction))then
       call stop_program('Direction size does not match array size in set_direction')
    end if
    this%direction = direction

  end subroutine set_direction
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module subroutine print_graph(this)
    implicit none
    class(array_type), intent(in) :: this

    print *, '--- Computation Graph Tree ---'
    call print_tree(this, '')
    print *, '--- End Graph ---'

  contains
    recursive subroutine print_tree(node, prefix)
      class(array_type), intent(in) :: node
      character(len=*), intent(in) :: prefix
      character(len=1024) :: new_prefix
      character(len=3) :: ownership_char
      ! ownership character
      character(len=20) :: node_addr

      write(node_addr, '(I0)') loc(node)
      print *, trim(prefix) // '└── [' // trim(node%operation) // &
           '] @' // trim(adjustl(node_addr))

      ! Print left operand
      if(associated(node%left_operand))then
         if(node%owns_left_operand)then
            ownership_char = '(*)'
         else
            ownership_char = ''
         end if
         if(associated(node%right_operand))then
            new_prefix = trim(prefix) // '    ├── L' // trim(ownership_char) // ': '
         else
            new_prefix = trim(prefix) // '    └── L' // trim(ownership_char) // ': '
         end if
         call print_tree(node%left_operand, new_prefix)
      end if

      ! Print right operand
      if(associated(node%right_operand))then
         if(node%owns_right_operand)then
            ownership_char = '(*)'
         else
            ownership_char = ''
         end if
         new_prefix = trim(prefix) // '    └── R' // trim(ownership_char) // ': '
         call print_tree(node%right_operand, new_prefix)
      end if
    end subroutine print_tree

  end subroutine print_graph
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function add_arrays(a, b) result(c)
    !! Add two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s

    ! Safely create result array
    c => a%create_result()
    if(b%is_sample_dependent)then
       c%val = a%val + b%val
    else
       do s = 1, size(a%val, 2)
          c%val(:,s) = a%val(:,s) + b%val(:,1)
       end do
    end if

    c%get_partial_left => get_partial_add
    c%get_partial_right => get_partial_add
    ! Set up computation graph
    if(a%requires_grad .or. b%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'add'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function add_arrays
!-------------------------------------------------------------------------------
  module function add_real2d(a, b) result(c)
    !! Add a real array to an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val + b

    c%get_partial_left => get_partial_add
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'add'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function add_real2d
!-------------------------------------------------------------------------------
  module function real2d_add(a, b) result(c)
    !! Add a real array to an autodiff array
    implicit none
    real(real32), dimension(:,:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => add_real2d(b, a)
  end function real2d_add
!-------------------------------------------------------------------------------
  module function add_real1d(a, b) result(c)
    !! Add a real array to an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s

    c => a%create_result()
    do concurrent(s=1:size(a%val,2))
       c%val(:,s) = a%val(:,s) + b(:)
    end do

    c%get_partial_left => get_partial_add
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'add'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function add_real1d
!-------------------------------------------------------------------------------
  module function real1d_add(a, b) result(c)
    !! Add a real array to an autodiff array
    implicit none
    real(real32), dimension(:), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => add_real1d(b, a)
  end function real1d_add
!-------------------------------------------------------------------------------
  module function add_scalar(a, b) result(c)
    !! Add a scalar to an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val + b

    c%get_partial_left => get_partial_add
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'add'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function add_scalar
!-------------------------------------------------------------------------------
  module function scalar_add(a, b) result(c)
    !! Add a scalar to an autodiff array
    implicit none
    real(real32), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => add_scalar(b, a)
  end function scalar_add
!-------------------------------------------------------------------------------
  function get_partial_add(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    output = upstream_grad
  end function get_partial_add
!###############################################################################


!###############################################################################
  module function subtract_arrays(a, b) result(c)
    !! Subtract two autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val - b%val

    c%get_partial_left => get_partial_add
    c%get_partial_right => get_partial_negate
    if(a%requires_grad .or. b%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'subtract'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function subtract_arrays
!-------------------------------------------------------------------------------
  module function subtract_real1d(a, b) result(c)
    !! Subtract a real array from an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s

    c => a%create_result()
    do concurrent(s=1:size(a%val,2))
       c%val(:,s) = a%val(:,s) - b(s)
    end do

    c%get_partial_left => get_partial_add
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'subtract_scalar'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function subtract_real1d
!-------------------------------------------------------------------------------
  module function subtract_scalar(a, b) result(c)
    !! Subtract a scalar from an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val - b

    c%get_partial_left => get_partial_add
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'subtract_scalar'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function subtract_scalar
!-------------------------------------------------------------------------------
  module function scalar_subtract(a, b) result(c)
    !! Subtract an autodiff array from a scalar
    implicit none
    real(real32), intent(in) :: a
    class(array_type), intent(in), target :: b
    type(array_type), pointer :: c

    c => negate_array(b)
    c%val = a + c%val

    c%get_partial_left => get_partial_negate
    if(b%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = b%is_forward
       c%operation = 'subtract_scalar'
       c%left_operand => b
       c%owns_left_operand = b%is_temporary
    end if
  end function scalar_subtract
!-------------------------------------------------------------------------------
  module function negate_array(a) result(c)
    !! Negate an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = -a%val

    c%get_partial_left => get_partial_negate
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'negate'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function negate_array
!-------------------------------------------------------------------------------
  function get_partial_negate(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => -upstream_grad
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_negate
!###############################################################################


!###############################################################################
  module function multiply_arrays(a, b) result(c)
    !! Multiply two autodiff arrays (element-wise)
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s

    if(b%is_scalar)then
       c => a%create_result()
       c%val = a%val * b%val(1,1)
    elseif(.not.b%is_sample_dependent)then
       do s = 1, size(a%val,2)
          c%val(:,s) = a%val(:,s) * b%val(:,1)
       end do
    elseif(size(a%val,1).ne.size(b%val,1).and.size(a%val,2).eq.size(b%val,2))then
       if(size(a%val,1) .eq. 1)then
          c => b%create_result()
          do concurrent(s=1:size(a%val,2))
             c%val(:,s) = a%val(1,s) * b%val(:,s)
          end do
       elseif(size(b%val,1) .eq. 1)then
          c => a%create_result()
          do concurrent(s=1:size(a%val,2))
             c%val(:,s) = a%val(:,s) * b%val(1,s)
          end do
       end if
    else
       c => a%create_result()
       c%val = a%val * b%val
    end if

    c%get_partial_left => get_partial_multiply_left
    c%get_partial_right => get_partial_multiply_right
    if(a%requires_grad .or. b%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'multiply'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function multiply_arrays
!-------------------------------------------------------------------------------
  module function multiply_scalar(a, scalar) result(c)
    !! Multiply autodiff array by scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    c => a%create_result()
    c%val = a%val * scalar

    c%get_partial_left => get_partial_multiply_left
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'multiply_scalar'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
    allocate(b_array)
    b_array%is_scalar = .true.
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%right_operand => b_array
    c%owns_right_operand = .true.
  end function multiply_scalar
!-------------------------------------------------------------------------------
  module function scalar_multiply(scalar, a) result(c)
    !! Multiply scalar by autodiff array
    implicit none
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => multiply_scalar(a, scalar)
  end function scalar_multiply
!-------------------------------------------------------------------------------
  module function multiply_logical(a, b) result(c)
    !! Multiply two logical arrays (element-wise)
    implicit none
    class(array_type), intent(in), target :: a
    logical, dimension(:,:), intent(in) :: b
    type(array_type), pointer :: c

    integer :: s, i

    c => a%create_result()
    do concurrent(s=1:size(a%val,2), i=1:size(a%val,1))
       if(b(i,s))then
          c%val(i,s) = a%val(i,s)
       else
          c%val(i,s) = 0.0_real32
       end if
    end do

    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'multiply_logical'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if

  end function multiply_logical
!-------------------------------------------------------------------------------
  function get_partial_multiply_left(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: right_is_temporary_local
    type(array_type), pointer :: ptr

    right_is_temporary_local = this%right_operand%is_temporary
    this%right_operand%is_temporary = .false.
    if(this%right_operand%is_scalar)then
       ptr => upstream_grad * this%right_operand%val(1,1)
    else
       ptr => upstream_grad * this%right_operand
    end if
    this%right_operand%is_temporary = right_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_multiply_left
!-------------------------------------------------------------------------------
  function get_partial_multiply_right(this, upstream_grad) result(output)
    !! Get partial derivative with respect to right operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    if(this%left_operand%is_scalar)then
       ptr => upstream_grad * this%left_operand%val(1,1)
    else
       ptr => upstream_grad * this%left_operand
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_multiply_right
!###############################################################################


!###############################################################################
  module function divide_arrays(a, b) result(c)
    !! Divide two autodiff arrays (element-wise)
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    integer :: s

    if(all(shape(a%val) .eq. shape(b%val)))then
       c => a%create_result()
       c%val = a%val / b%val
    elseif(size(a%val,1).ne.size(b%val,1).and.size(a%val,2).eq.size(b%val,2))then
       if(size(a%val,1) .eq. 1)then
          c => b%create_result()
          do concurrent(s=1:size(a%val,2))
             c%val(:,s) = a%val(1,s) / b%val(:,s)
          end do
       elseif(size(b%val,1) .eq. 1)then
          c => a%create_result()
          do concurrent(s=1:size(a%val,2))
             c%val(:,s) = a%val(:,s) / b%val(1,s)
          end do
       end if
    end if

    c%get_partial_left => get_partial_divide_left
    c%get_partial_right => get_partial_divide_right
    if(a%requires_grad .or. b%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'divide'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function divide_arrays
!-------------------------------------------------------------------------------
  module function divide_scalar(a, scalar) result(c)
    !! Divide autodiff array by scalar
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    c => a%create_result()
    c%val = a%val / scalar

    c%get_partial_left => get_partial_divide_left
    c%get_partial_right => get_partial_divide_right
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'divide_scalar'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
    allocate(b_array)
    b_array%is_scalar = .true.
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%right_operand => b_array
    c%owns_right_operand = .true.
  end function divide_scalar
!-------------------------------------------------------------------------------
  module function scalar_divide(scalar, a) result(c)
    !! Divide scalar by autodiff array
    implicit none
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    c => a%create_result()
    c%val = scalar / a%val

    c%get_partial_left => get_partial_divide_left
    c%get_partial_right => get_partial_divide_right
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'scalar_divide'
       c%right_operand => a
       c%owns_right_operand = a%is_temporary
    end if
    allocate(b_array)
    b_array%is_scalar = .true.
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%left_operand => b_array
    c%owns_left_operand = .true.
  end function scalar_divide
!-------------------------------------------------------------------------------
  module function divide_real1d(a, b) result(c)
    !! Divide autodiff array by a real array
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), dimension(:), intent(in) :: b
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    integer :: s

    c => a%create_result()
    do concurrent(s=1:size(a%val,2))
       c%val(:,s) = a%val(:,s) / b(s)
    end do

    c%get_partial_left => get_partial_divide_left
    c%get_partial_right => get_partial_divide_right
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'divide_real1d'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
    allocate(b_array)
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[1, size(b)])
    b_array%val(1,:) = b
    c%right_operand => b_array
    c%owns_right_operand = .true.
  end function divide_real1d
!-------------------------------------------------------------------------------
  function get_partial_divide_left(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: right_is_temporary_local
    type(array_type), pointer :: ptr

    right_is_temporary_local = this%right_operand%is_temporary
    this%right_operand%is_temporary = .false.
    if(this%right_operand%is_scalar)then
       ptr => upstream_grad / this%right_operand%val(1,1)
    else
       ptr => upstream_grad / this%right_operand
    end if
    this%right_operand%is_temporary = right_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_divide_left
!-------------------------------------------------------------------------------
  function get_partial_divide_right(this, upstream_grad) result(output)
    !! Get partial derivative with respect to right operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local, right_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    right_is_temporary_local = this%right_operand%is_temporary
    this%left_operand%is_temporary = .false.
    this%right_operand%is_temporary = .false.
    if(this%left_operand%is_scalar)then
       ptr => (-upstream_grad * this%left_operand%val(1,1)) / &
            (this%right_operand * this%right_operand)
    else
       ptr => (-upstream_grad * this%left_operand) / &
            (this%right_operand * this%right_operand)
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    this%right_operand%is_temporary = right_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_divide_right
!###############################################################################


!###############################################################################
  module function power_arrays(a, b) result(c)
    !! Raise autodiff array to power of another array
    implicit none
    class(array_type), intent(in), target :: a, b
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = a%val ** b%val

    c%get_partial_left => get_partial_power_base
    c%get_partial_right => get_partial_power_exponent
    if(a%requires_grad .or. b%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward .or. b%is_forward
       c%operation = 'power'
       c%left_operand => a
       c%right_operand => b
       c%owns_left_operand = a%is_temporary
       c%owns_right_operand = b%is_temporary
    end if
  end function power_arrays
!-------------------------------------------------------------------------------
  module function power_real_scalar(a, scalar) result(c)
    !! Raise autodiff array to scalar power
    implicit none
    class(array_type), intent(in), target :: a
    real(real32), intent(in) :: scalar
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    c => a%create_result()
    c%val = a%val ** scalar

    c%get_partial_left => get_partial_power_base
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'power_scalar'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
    allocate(b_array)
    b_array%is_scalar = .true.
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%right_operand => b_array
    c%owns_right_operand = .true.

  end function power_real_scalar
!-------------------------------------------------------------------------------
  module function power_int_scalar(a, scalar) result(c)
    !! Raise autodiff array to scalar power
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: scalar
    type(array_type), pointer :: c

    c => power_real_scalar(a, real(scalar, real32))
  end function power_int_scalar
!-------------------------------------------------------------------------------
  module function scalar_power(scalar, a) result(c)
    implicit none
    real(real32), intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c
    type(array_type), pointer :: b_array

    c => a%create_result()
    c%val = scalar ** a%val

    c%get_partial_left => get_partial_power_base
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'scalar_power'
       c%right_operand => a
       c%owns_right_operand = a%is_temporary
    end if
    allocate(b_array)
    b_array%is_scalar = .true.
    b_array%is_sample_dependent = .false.
    b_array%requires_grad = .false.
    call b_array%allocate(array_shape=[1, 1])
    b_array%val(1, 1) = scalar
    c%left_operand => b_array
    c%owns_left_operand = .true.

  end function scalar_power
!-------------------------------------------------------------------------------
  module function int_scalar_power(scalar, a) result(c)
    implicit none
    integer, intent(in) :: scalar
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => scalar_power(real(scalar, real32), a)
  end function int_scalar_power
!-------------------------------------------------------------------------------
  function get_partial_power_base(this, upstream_grad) result(output)
    !! Get the partial gradient with respect to the base of the power operation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local, right_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    right_is_temporary_local = this%right_operand%is_temporary
    this%left_operand%is_temporary = .false.
    this%right_operand%is_temporary = .false.
    if(all(abs(this%right_operand%val - 1._real32).lt.1.E-6_real32))then
       output = upstream_grad
       return
    elseif(all(abs(this%right_operand%val - 2._real32).lt.1.E-6_real32))then
       ptr => upstream_grad * this%left_operand * 2._real32
    else
       if(this%right_operand%is_scalar)then
          ptr => upstream_grad * this%right_operand%val(1,1) * &
               this%left_operand ** ( this%right_operand%val(1,1) - 1.0_real32 )
       else
          ptr => upstream_grad * this%right_operand * &
               this%left_operand ** ( this%right_operand - 1.0_real32 )
       end if
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    this%right_operand%is_temporary = right_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_power_base
!-------------------------------------------------------------------------------
  function get_partial_power_exponent(this, upstream_grad) result(output)
    !! Get the partial gradient with respect to the exponent of the power operation
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local, this_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this_is_temporary_local = this%is_temporary
    this%left_operand%is_temporary = .false.
    this%is_temporary = .false.
    if(this%left_operand%is_scalar)then
       ptr => upstream_grad * log(this%left_operand%val(1,1)) * this
    else
       ptr => upstream_grad * log(this%left_operand) * this
    end if
    this%left_operand%is_temporary = left_is_temporary_local
    this%is_temporary = this_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_power_exponent
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!


!###############################################################################
  module function exp_array(a) result(c)
    !! Exponential function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = exp(a%val)

    c%get_partial_left => get_partial_exp
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'exp'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function exp_array
!-------------------------------------------------------------------------------
  function get_partial_exp(this, upstream_grad) result(output)
    !! Get partial derivative with respect to left operand
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: this_is_temporary_local
    type(array_type), pointer :: ptr

    this_is_temporary_local = this%is_temporary
    this%is_temporary = .false.
    ptr => upstream_grad * this
    this%is_temporary = this_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_exp
!###############################################################################


!###############################################################################
  module function log_array(a) result(c)
    !! Natural logarithm function for autodiff arrays
    implicit none
    class(array_type), intent(in), target :: a
    type(array_type), pointer :: c

    c => a%create_result()
    c%val = log(a%val)

    c%get_partial_left => get_partial_log
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'log'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
  end function log_array
!-------------------------------------------------------------------------------
  function get_partial_log(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    logical :: left_is_temporary_local
    type(array_type), pointer :: ptr

    left_is_temporary_local = this%left_operand%is_temporary
    this%left_operand%is_temporary = .false.
    ptr => upstream_grad / this%left_operand
    this%left_operand%is_temporary = left_is_temporary_local
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_log
!###############################################################################


!##############################################################################!
! * * * * * * * * * * * * * * * * * * *  * * * * * * * * * * * * * * * * * * * !
!##############################################################################!



!###############################################################################
  module function mean_array(a, dim) result(c)
    !! Compute mean values along a dimension
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    type(array_type), pointer :: c

    integer :: s
    real(real32) :: rtmp1

    ! if(size(a%shape) .ne. 1)then
    !    call stop_program("mean_array: only 1D arrays can be used")
    ! end if

    if(dim.eq.1)then
       c => a%create_result(array_shape = [1, size(a%val,2)])
       rtmp1 = real(size(a%val,1), real32)
       do concurrent(s=1:size(a%val,2))
          c%val(1,s) = sum(a%val(:,s)) / rtmp1
       end do
    else if(dim.eq.2)then
       c => a%create_result(array_shape = [a%shape, 1])
       rtmp1 = real(size(a%val,2), real32)
       do concurrent(s=1:size(a%val,1))
          c%val(s,1) = sum(a%val(s,:)) / rtmp1
       end do
       c%is_sample_dependent = .false.
    else
       call stop_program("mean_array: only 1 or 2 dimensions are supported")
    end if
    c%indices = [dim, 1]

    c%get_partial_left => get_partial_mean
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'mean_array'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if

  end function mean_array
!-------------------------------------------------------------------------------
  function get_partial_mean(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output

    real(real32) :: rtmp1
    type(array_type), pointer :: ptr

    ! Calculate the number of elements that were averaged
    rtmp1 = real(size(this%left_operand%val, this%indices(1)), real32)

    if(this%is_forward)then
       ptr => sum( upstream_grad, dim = this%indices(1) ) / rtmp1
    else
       ptr => spread( &
            upstream_grad / rtmp1, &
            dim=this%indices(1), &
            index=this%indices(2), &
            ncopies= size(this%left_operand%val, this%indices(1)) &
       )
    end if
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_mean
!###############################################################################


!###############################################################################
  module function sum_array(a, dim) result(c)
    !! Sum values along a dimension
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    type(array_type), pointer :: c

    integer :: s

    if(dim.eq.1)then
       c => a%create_result(array_shape=[1, size(a%val,2)])
       do concurrent(s=1:size(a%val,2))
          c%val(1,s) = sum(a%val(:,s))
       end do
    else if(dim.eq.2)then
       c => a%create_result(array_shape=[a%shape, 1])
       do concurrent(s=1:size(a%val,1))
          c%val(s,1) = sum(a%val(s,:))
       end do
       c%is_sample_dependent = .false.
    else
       call stop_program("sum_array: only 1 or 2 dimensions are supported")
    end if
    c%indices = [dim, 1]

    c%get_partial_left => get_partial_sum_reverse
    c%get_partial_right => get_partial_sum_forward
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'sum_array'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if

  end function sum_array
!-------------------------------------------------------------------------------
  module function sum_array_output_array(a, dim, new_dim_index, new_dim_size) result(c)
    !! Sum values along a dimension and return an autodiff array
    implicit none
    class(array_type), intent(in), target :: a
    integer, intent(in) :: dim
    integer, intent(in) :: new_dim_index
    integer, intent(in) :: new_dim_size
    type(array_type), pointer :: c

    if(size(a%shape) .ne. 1)then
       call stop_program("sum_array_output_array: only 1D arrays can be used")
    end if

    allocate(c)
    ! sum 1D array by using shape to swap dimensions
    if(dim.eq.1)then
       call c%allocate(array_shape=[new_dim_size, size(a%val,2)])
       c%val = 0.0_real32
       c%val(new_dim_index,:) = sum(a%val(:,:), dim=1)
    else if(dim.eq.2)then
       call c%allocate(array_shape=[size(a%val,1), new_dim_size])
       c%val = 0.0_real32
       c%val(:,new_dim_index) = sum(a%val(:,:), dim=2)
    end if

    c%get_partial_left => get_partial_sum_reverse
    c%is_sample_dependent = a%is_sample_dependent
    if(a%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = a%is_forward
       c%operation = 'sum_array_output_array'
       c%left_operand => a
       c%owns_left_operand = a%is_temporary
    end if
    c%indices = [dim, new_dim_index]
  end function sum_array_output_array
!-------------------------------------------------------------------------------
  function get_partial_sum_reverse(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => spread( &
         upstream_grad, &
         dim=this%indices(1), &
         index=this%indices(2), &
         ncopies= size(this%left_operand%val, this%indices(1)) &
    )
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_sum_reverse
!-------------------------------------------------------------------------------
  function get_partial_sum_forward(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => sum( &
         upstream_grad, &
         dim = this%indices(1) &
    )
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_sum_forward
!###############################################################################


!###############################################################################
  module function spread_array(source, dim, index, ncopies) result(c)
    !! Spread an autodiff array along a dimension
    implicit none
    class(array_type), intent(in), target :: source
    integer, intent(in) :: dim
    integer, intent(in) :: index
    integer, intent(in) :: ncopies
    type(array_type), pointer :: c

    integer :: s

    if(size(source%shape) .ne. 1)then
       call stop_program("spread: only 1D arrays can be used")
    end if

    if(dim.eq.1)then
       c => source%create_result(array_shape=[ncopies, size(source%val,2)])
       do concurrent(s=1:ncopies)
          c%val(s, :) = source%val(index, :)
       end do
    else if(dim.eq.2)then
       c => source%create_result(array_shape=[size(source%val,1), ncopies])
       do concurrent(s=1:ncopies)
          c%val(:, s) = source%val(:, index)
       end do
    else
       call stop_program("spread: only 1 or 2 dimensions are supported")
    end if
    c%indices = [index]
    allocate(c%adj_ja(2,1))
    c%adj_ja(:,1) = [ dim, size(source%val,dim) ]

    c%get_partial_left => get_partial_spread
    if(source%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = source%is_forward
       c%operation = 'spread'
       c%left_operand => source
       c%owns_left_operand = source%is_temporary
    end if
  end function spread_array
!-------------------------------------------------------------------------------
  function get_partial_spread(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => unspread( &
         upstream_grad, &
         this%indices(1), &
         this%adj_ja(1,1), &
         this%adj_ja(2,1) &
    )
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_spread
!###############################################################################


!###############################################################################
  module function unspread_array(source, index, dim, new_size) result(c)
    !! Unpack an autodiff array
    implicit none
    class(array_type), intent(in), target :: source
    integer, intent(in) :: index
    integer, intent(in) :: new_size, dim
    type(array_type), pointer :: c

    integer :: i, s


    if(dim.eq.1)then
       c => source%create_result(array_shape = [ new_size, size(source%val,2) ])
       c%val = 0.0_real32
       do concurrent(i=1:size(source%val,1), s=1:size(source%val,2))
          c%val(index,s) = c%val(index,s) + source%val(i,s)
       end do
    elseif(dim.eq.2)then
       c => source%create_result( array_shape = [ size(source%val,1), new_size ] )
       c%val = 0.0_real32
       do concurrent(i=1:size(source%val,1), s=1:size(source%val,2))
          c%val(i,index) = c%val(i,index) + source%val(i,s)
       end do
    end if
    c%indices = [index]
    allocate(c%adj_ja(2,1))
    c%adj_ja(:,1) = [ dim, new_size ]

    c%get_partial_left => get_partial_unspread
    if(source%requires_grad)then
       c%requires_grad = .true.
       c%is_forward = source%is_forward
       c%operation = 'unspread'
       c%left_operand => source
       c%owns_left_operand = source%is_temporary
    end if
  end function unspread_array
!-------------------------------------------------------------------------------
  function get_partial_unspread(this, upstream_grad) result(output)
    implicit none
    class(array_type), intent(inout) :: this
    type(array_type), intent(in) :: upstream_grad
    type(array_type) :: output
    type(array_type), pointer :: ptr

    ptr => spread( &
         upstream_grad, &
         this%indices(1), &
         this%adj_ja(1,1), &
         this%adj_ja(2,1) &
    )
    call output%assign_and_deallocate_source(ptr)
  end function get_partial_unspread
!###############################################################################

end submodule diffstruc__types_submodule
