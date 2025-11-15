program test_functions
  !! Example demonstrating second-order derivative computation using your autodiff system
  use coreutils, only: real32
  use diffstruc
  implicit none

  ! Variables
  type(array_type) :: x, y
  type(array_type), pointer :: f, hessian1, hessian2, grad1, grad2, &
       du_dx, d2u_dx2, residual, loss
  integer :: i, j

  write(*,*) "=== Second-Order Derivatives Example ==="

  ! Create a 2D input vector x = [x1, x2]
  call x%allocate(array_shape=[2, 1])
  call y%allocate(array_shape=[2, 1])
  x%is_temporary = .false.
  y%is_temporary = .false.
  x%val(1, 1) = 1.0_real32
  x%val(2, 1) = 5.0_real32
  y%val(1, 1) = 3.0_real32
  y%val(2, 1) = 4.0_real32
  call x%set_requires_grad(.true.)
  call y%set_requires_grad(.true.)

  write(*,*) "Computing function: f"
  write(*,*) "At point: x =", x%val(1,1)

  f => pack(x, [1], 1) * pack(x, [2], 1) **2 + pack(x, [1], 1)**2
  f%is_temporary = .false.

  write(*,*) "Function value f =", f%val(1,1)

  ! Compute first derivatives (gradient)
  call x%set_direction([1._real32, 0._real32])
  grad1 => f%grad_forward(x)
  grad1%is_temporary = .false.
  call x%set_direction([0._real32, 1._real32])
  grad2 => f%grad_forward(x)
  grad2%is_temporary = .false.
  write(*,*) "First derivatives (gradient):"
  write(*,*) "  df/dx1 =", grad1%val(1,1)
  write(*,*) "  df/dx2 =", grad2%val(1,1)

  ! Compute second derivatives (Hessian)
  write(*,*) "Computing Hessian matrix..."
  call x%set_direction([1._real32, 1._real32])
  call grad1%grad_reverse( reset_graph=.true. )
  if(associated(x%grad)) then
     hessian1 => x%grad
     write(*,*) "  d^2f/dx1dx1 =", hessian1%val(1,1)
     write(*,*) "  d^2f/dx1dx2 =", hessian1%val(2,1)
  end if

  call grad2%grad_reverse( reset_graph=.true. )
  if(associated(x%grad)) then
     hessian2 => x%grad
     write(*,*) "  d^2f/dx2dx1 =", hessian2%val(1,1)
     write(*,*) "  d^2f/dx2dx2 =", hessian2%val(2,1)
  end if

  call grad1%nullify_graph(ignore_ownership=.false.)
  call grad2%nullify_graph(ignore_ownership=.true.)
  write(*,*) "=== Example 1 Complete ==="


  f => x ** 4._real32 + x ** 2 * y
  f%is_temporary = .false.

  write(*,*) "Function value f =", f%val(1,1)

  ! Compute first derivatives (gradient)
  call x%set_direction([1._real32, 1._real32])
  grad1 => f%grad_forward(x)
  grad1%is_temporary = .false.
  write(*,*) "First derivatives (gradient):"
  write(*,*) "  df/dx1 =", grad1%val(1,1)
  write(*,*) "  df/dx2 =", grad1%val(2,1)
  call y%set_direction([1._real32, 1._real32])
  grad2 => f%grad_forward(y)
  grad2%is_temporary = .false.
  write(*,*) "  df/dy1 =", grad2%val(1,1)
  write(*,*) "  df/dy2 =", grad2%val(2,1)

  ! Compute second derivatives (Hessian)
  write(*,*) "Computing Hessian matrix..."
  call x%set_direction([1._real32, 1._real32])
  hessian1 => grad1%grad_forward(x)
  write(*,*) "  d^2f/dx^2 =", hessian1%val(1,1)
  write(*,*) "  d^2f/dx^2 =", hessian1%val(2,1)

  call y%set_direction([1._real32, 1._real32])
  hessian2 => grad2%grad_forward(x)
  write(*,*) "  d^2f/dydx =", hessian2%val(1,1)
  write(*,*) "  d^2f/dydx =", hessian2%val(2,1)

  call grad1%nullify_graph(ignore_ownership=.true.)
  call grad2%nullify_graph(ignore_ownership=.false.)
  call hessian1%nullify_graph(ignore_ownership=.true.)
  call hessian2%nullify_graph(ignore_ownership=.true.)
  write(*,*) "=== Example 2 Complete ==="


  ! Compute second derivatives of a more complex function
  call f%reset_graph()
  f => mean( x * tanh(x), dim = 1 )
  f%is_temporary = .false.
  write(*,*) "Function value f =", f%val(:,1)
  call x%set_direction([1._real32, 0._real32])
  grad1 => f%grad_forward(x)
  grad1%is_temporary = .false.
  write(*,*) "Gradient (first derivatives):"
  write(*,*) "  df/dx1 =", grad1%val(:,1)
  call x%set_direction([1._real32, 0._real32])
  hessian1 => grad1%grad_forward(x)
  write(*,*) "Hessian matrix:"
  write(*,*) "  d^2f/dx1dx1 =", hessian1%val(:,1)

  call grad1%nullify_graph(ignore_ownership=.true.)
  call hessian1%nullify_graph(ignore_ownership=.true.)
  write(*,*) "=== Example 3 Complete ==="


  ! Compute first and second derivatives using forward mode
  call f%reset_graph()
  f => x * x * x
  f%is_temporary = .false.
  write(*,*) "Function value f =", f%val(:,1)
  call x%set_direction([1._real32, 1._real32])
  grad1 => f%grad_forward(x)
  write(*,*) "Gradient (first derivatives):"
  write(*,*) "  df/dx1 =", grad1%val(:,1)
  hessian1 = grad1%grad_forward(x)
  write(*,*) "Hessian matrix:"
  write(*,*) "  d^2f/dx1dx1 =", hessian1%val(:,1)

  call grad1%nullify_graph(ignore_ownership=.true.)
  call hessian1%nullify_graph(ignore_ownership=.false.)
  write(*,*) "=== Example 4 Complete ==="


  ! Compute derivatives for residual function and loss
  call f%reset_graph()
  f => tanh(x)
  f%is_temporary = .false.

  du_dx  => f%grad_forward(x)                     ! ∂u/∂x
  d2u_dx2 => du_dx%grad_forward(x)                 ! ∂²u/∂x²
  write(*,*) "  df/dx1 =", du_dx%val(:,1)
  write(*,*) "  d²f/dx1² =", d2u_dx2%val(:,1)

  residual => d2u_dx2 - 12._real32 * x * x - 2._real32 * y
  write(*,*) "  Residual:", residual%val(:,1)

  ! allocate(loss)
  loss => sum(residual ** 2, dim=1)
  write(*,*) "  Loss:", loss%val(1,1)

  call loss%grad_reverse()

  write(*,*) "Gradient of loss:"
  if(associated(x%grad)) then
     write(*,*) "  dLoss/dx1 =", x%grad%val(1,1)
     write(*,*) "  dLoss/dx2 =", x%grad%val(2,1)
  end if

  call loss%nullify_graph(ignore_ownership=.false.)
  write(*,*) "=== Example 5 Complete ==="


  ! Compute first and second derivatives using forward mode
  call f%reset_graph()
  f => tanh(x**2)
  f%is_temporary = .false.

  du_dx  => f%grad_forward(x)                     ! ∂u/∂x
  d2u_dx2 => du_dx%grad_forward(x)                 ! ∂²u/∂x²
  write(*,*) "  df/dx1 =", du_dx%val(:,1)
  write(*,*) "  d²f/dx1² =", d2u_dx2%val(:,1)

  call du_dx%nullify_graph(ignore_ownership=.false.)
  call d2u_dx2%nullify_graph(ignore_ownership=.false.)
  write(*,*) "=== Example 6 Complete ==="


  call f%reset_graph()
  f => dot_product(x, y)
  f%is_temporary = .false.
  write(*,*) "Function value f =", f%val(1,1)
  call f%grad_reverse( reset_graph=.true. )
  if(associated(x%grad)) then
     write(*,*) "Gradient w.r.t x:"
     write(*,*) "  df/dx1 =", x%grad%val(1,1)
     write(*,*) "  df/dx2 =", x%grad%val(2,1)
  end if
  if(associated(y%grad)) then
     write(*,*) "Gradient w.r.t y:"
     write(*,*) "  df/dy1 =", y%grad%val(1,1)
     write(*,*) "  df/dy2 =", y%grad%val(2,1)
  end if

  call f%nullify_graph(ignore_ownership=.true.)
  write(*,*) "=== Example 7 Complete ==="

end program test_functions
