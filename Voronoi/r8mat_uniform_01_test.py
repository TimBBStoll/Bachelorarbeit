# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import platform
import r8mat_uniform_01
def r8mat_uniform_01_test ( ):



  m = 5
  n = 4
  seed = 123456789

  print ( '' )
  print ( 'R8MAT_UNIFORM_01_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  R8MAT_UNIFORM_01 computes a random R8MAT.' )
  print ( '' )
  print ( '  0 <= X <= 1' )
  print ( '  Initial seed is %d' % ( seed ) )

  v, seed = r8mat_uniform_01 ( m, n, seed )

  print ( '' )
  print ( '  Random R8MAT:' )
  print ( '' )
  for i in range ( 0, m ):
    for j in range ( 0, n ):
      print ( '  %8.4g' % ( v[i,j] ), end = '' )
    print ( '' )
#
#  Terminate.
#
  print ( '' )
  print ( 'R8MAT_UNIFORM_01_TEST:' )
  print ( '  Normal end of execution.' )
  return