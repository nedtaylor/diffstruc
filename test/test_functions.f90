program test_functions
  !! Example demonstrating second-order derivative computation using your autodiff system
  use corestruc, only: real32
  use diffstruc
  implicit none

  ! Variables
  type(array_type) :: x, y, f, f_x, residual, loss
  type(array_type) :: hessian, grad
  type(array_type), pointer :: input, du_dx, d2u_dx2
  integer :: i, j

  write(*,*) "=== Second-Order Derivatives Example ==="

  ! Create a 2D input vector x = [x1, x2]
  call x%allocate(array_shape=[2, 1])
  call y%allocate(array_shape=[2, 1])
  x%val(1, 1) = 1.0_real32
  x%val(2, 1) = 5.0_real32
  y%val(1, 1) = 3.0_real32
  y%val(2, 1) = 4.0_real32
  call x%set_requires_grad(.true.)

  write(*,*) "Computing function: f"
  write(*,*) "At point: x =", x%val(1,1)

  f = pack(x, [1], 1) * pack(x, [2], 1) **2 + pack(x, [1], 1)**2

  write(*,*) "Function value f =", f%val(1,1)

  ! Compute first derivatives (gradient)
  call x%set_direction([1._real32, 1._real32])
  call f%grad_reverse( record_graph=.true.)
  write(*,*) "First derivatives (gradient):"
  if(associated(x%grad)) then
     write(*,*) "  df/dx1 =", x%grad%val(1,1)
     write(*,*) "  df/dx2 =", x%grad%val(2,1)
  end if

  ! Compute second derivatives (Hessian)
  write(*,*) "Computing Hessian matrix..."
  call x%set_direction([1._real32, 0._real32])
  hessian = x%grad%grad_forward(x)
  write(*,*) "  d^2f/dx1dx1 =", hessian%val(1,1)
  write(*,*) "  d^2f/dx1dx2 =", hessian%val(2,1)

  call x%set_direction([0._real32, 1._real32])
  hessian = x%grad%grad_forward(x)
  write(*,*) "  d^2f/dx2dx1 =", hessian%val(1,1)
  write(*,*) "  d^2f/dx2dx2 =", hessian%val(2,1)

  write(*,*) "=== Example 1 Complete ==="


  call x%set_requires_grad(.true.)
  call y%set_requires_grad(.true.)

  f = x ** 4._real32 + x ** 2 * y

  write(*,*) "Function value f =", f%val(1,1)

  ! Compute first derivatives (gradient)
  call x%set_direction([1._real32, 1._real32])
  call f%grad_reverse( record_graph=.true., reset_graph=.true. )
  write(*,*) "First derivatives (gradient):"
  if(associated(x%grad)) then
     write(*,*) "  df/dx1 =", x%grad%val(1,1)
     write(*,*) "  df/dx2 =", x%grad%val(2,1)
  end if
  if(associated(y%grad)) then
     write(*,*) "  df/dy1 =", y%grad%val(1,1)
     write(*,*) "  df/dy2 =", y%grad%val(2,1)
  end if

  ! Compute second derivatives (Hessian)
  write(*,*) "Computing Hessian matrix..."
  call x%set_direction([1._real32, 1._real32])
  hessian = x%grad%grad_forward(x)
  write(*,*) "  d^2f/dx^2 =", hessian%val(1,1)
  write(*,*) "  d^2f/dx^2 =", hessian%val(2,1)

  hessian = y%grad%grad_forward(x)
  write(*,*) "  d^2f/dydx =", hessian%val(1,1)
  write(*,*) "  d^2f/dydx =", hessian%val(2,1)

  write(*,*) "=== Example 2 Complete ==="


  ! Compute gradients using forward mode
  call f%reset_graph()
  f = x ** 4._real32 + x ** 2 * y
  write(*,*) "Function value f =", f%val(1,1)
  grad = f%grad_forward(x)
  write(*,*) "Gradient (first derivatives):"
  write(*,*) "  df/dx1 =", grad%val(1,1)
  write(*,*) "  df/dx2 =", grad%val(2,1)
  grad = f%grad_forward(y)
  write(*,*) "  df/dy1 =", grad%val(1,1)
  write(*,*) "  df/dy2 =", grad%val(2,1)

  write(*,*) "=== Example 3 Complete ==="

  call f%reset_graph()
  f = mean( x * tanh(x), dim = 1 )
  write(*,*) "Function value f =", f%val(:,1)
  call f%grad_reverse( record_graph=.true. )
  write(*,*) "Gradient (first derivatives):"
  if(associated(x%grad)) then
     write(*,*) "  df/dx1 =", x%grad%val(:,1)
  end if
  hessian = x%grad%grad_forward(x)
  write(*,*) "Hessian matrix:"
  write(*,*) "  d^2f/dx1dx1 =", hessian%val(:,1)

  write(*,*) "=== Example 4 Complete ==="

  call f%reset_graph()
!   f = x**4
  f = x * x * x!* x * x!tanh(x) !mean( x * tanh(x), dim = 1 )
  write(*,*) "Function value f =", f%val(:,1)
  call f%grad_reverse( record_graph=.true. )
  write(*,*) "Gradient (first derivatives):"
  if(associated(x%grad)) then
     write(*,*) "  df/dx1 =", x%grad%val(:,1)
  end if
  x%id = 1
  hessian = x%grad%grad_forward(x)
  write(*,*) hessian%val(:,1)

  call f%reset_graph()
  f = tanh(x)! * x * x!* x * x!tanh(x) !mean( x * tanh(x), dim = 1 )

  du_dx  => f%grad_forward(x)                     ! ∂u/∂x
  d2u_dx2 => du_dx%grad_forward(x)                 ! ∂²u/∂x²
  write(*,*) "  df/dx1 =", du_dx%val(:,1)
  write(*,*) "  d²f/dx1² =", d2u_dx2%val(:,1)

  residual = d2u_dx2 - 12._real32 * x * x - 2._real32 * y
  write(*,*) "  Residual (should be close to 0):", residual%val(:,1)

  ! allocate(loss)
  loss = sum(residual ** 2, dim=1)
  write(*,*) "  Loss (should be close to 0):", loss%val(1,1)

  call loss%grad_reverse( record_graph=.true. )

  write(*,*) "Gradient of loss:"
  if(associated(x%grad)) then
     write(*,*) "  dLoss/dx1 =", x%grad%val(1,1)
     write(*,*) "  dLoss/dx2 =", x%grad%val(2,1)
  end if

  call f%reset_graph()
  f = tanh(x**2)! * x * x!* x * x!tanh(x) !mean( x * tanh(x), dim = 1 )

  du_dx  => f%grad_forward(x)                     ! ∂u/∂x
  d2u_dx2 => du_dx%grad_forward(x)                 ! ∂²u/∂x²
  write(*,*) "  df/dx1 =", du_dx%val(:,1)
  write(*,*) "  d²f/dx1² =", d2u_dx2%val(:,1)

end program test_functions
