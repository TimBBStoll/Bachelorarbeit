# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



def cvt_2d_sampling_H ( g_num = 0, it_num = 0, s1d_num = 0 ):


  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.spatial import Delaunay
  from scipy.spatial import Voronoi
  from scipy.spatial import voronoi_plot_2d

  from Halton_korrekt import halton







  if ( g_num <= 0 ):
    prompt = '  Enter number of generators:  '
    inp = input( prompt )
    g_num = int ( inp )

  if ( it_num <= 0 ):
    prompt = '  Enter number of iterations:  '
    inp = input ( prompt )
    it_num = int ( inp )

  if ( s1d_num <= 0 ):
    prompt = '  Enter number of sample points:  '
    inp = input ( prompt )
    s1d_num = int ( inp )

  s_num = s1d_num * s1d_num


#
#  Initialize the generators.
  x = halton(10, g_num)
  x_p = x[:, 0]
  y_p = x[:, 1]

  gx= x_p
  gy= y_p





#
  s_1d = np.linspace ( 0.0, 1.0, s1d_num )
  sx = np.zeros ( s_num )
  sy = np.zeros ( s_num )
  s = np.zeros ( [ s_num, 2 ] )
  k = 0
  for j in range ( 0, s1d_num ):
    for i in range ( 0, s1d_num ):
      sx[k] = s_1d[i]
      sy[k] = s_1d[j]
      s[k,0] = s_1d[i]
      s[k,1] = s_1d[j]
      k = k + 1
#

#
  step = np.zeros ( it_num )
  e = 1.0E-10 * np.ones ( it_num )
  gm = 1.0E-10 * np.ones ( it_num )

  for it in range ( 0, it_num ):

    step[it] = it

    global g
    g = np.zeros ( [ g_num, 2 ] )
    g[:,0] = gx[:]
    g[:,1] = gy[:]
    tri = Delaunay ( g )


#
    subfig1 = plt.subplot(2, 2, 1)
    global vor
    vor = Voronoi ( g )
    voronoi_plot_2d (vor, ax = subfig1, show_vertices=False, point_size=1, )
    #plt.show()

#

#
    subfig2 = plt.subplot ( 2, 2, 2 )
    plt.triplot ( gx, gy, tri.simplices.copy( ) )
    plt.plot ( gx, gy, 'o' )

#
#  For each sample point, find K, the index of the nearest generator.
#
    k = [ np.argmin ( [ np.inner ( gg - ss, gg - ss ) for gg in g ] ) for ss in s ]
#
#  For each generator, M counts the number of sample points it is nearest to.
#  Note that for a nonuniform density, we just set W to the density.
#
    w = np.ones ( s_num )
    m = np.bincount ( k, weights = w )
#
#  G is the average of the sample points it is nearest to.
#  Where M is zero, we shouldn't modify G.
#
    gx_new = np.bincount ( k, weights = sx )
    gy_new = np.bincount ( k, weights = sy )

    for i in range ( 0, g_num ):
      if ( 0 < m[i] ):
        gx_new[i] = gx_new[i] / float ( m[i] )
        gy_new[i] = gy_new[i] / float ( m[i] )
#
#  Compute the energy.
#
    e[it] = 0.0
    for i in range ( 0, s_num ):
      e[it] = e[it] + ( sx[i] - gx_new[k[i]] ) ** 2 \
                    + ( sy[i] - gy_new[k[i]] ) ** 2
#
#  Display the log of the energy.
#
    subfig3 = plt.subplot ( 2, 2, 3 )
    plt.plot ( step[0:it+1], np.log ( e[0:it+1] ), 'm-*', linewidth = 2 )
    plt.title ( 'Log (Energy)' )
    plt.xlabel ( 'Step' )
    plt.ylabel ( 'Energy' )
    plt.grid ( True )

#
#  Compute the generator motion.
#
    for i in range ( 0, g_num ):
      gm[it] = gm[it] + ( gx_new[i] - gx[i] ) ** 2 \
                      + ( gy_new[i] - gy[i] ) ** 2
#
#  Display the generator motion.
#
    subfig4 = plt.subplot ( 2, 2, 4 )
    plt.plot ( step[0:it+1], np.log ( gm[0:it+1] ), 'm-*', linewidth = 2 )
    plt.title ( 'Log (Average generator motion)' )
    plt.xlabel ( 'Step' )
    plt.ylabel ( 'Energy' )
    plt.grid ( True )

#
#  Put a title on the plot.
#
    super_title = 'Iteration ' + str ( it )
    plt.suptitle ( super_title )
#
#  Save the first and last plots.
#
    if ( it == 0 ):
      filename = 'initialk.png'
      plt.savefig ( filename )
    elif ( it == it_num - 1 ):
      filename = 'finalk.png'
      plt.savefig ( filename )




#
#  Show the plot.
#  Apparently, SHOW can only come AFTER SAVEFIG, or you save nothing.
#
    #plt.show ( )
#
#  Clear the figure.
#
    plt.clf ( )
#
#  Update the generators.

    gx = gx_new
    gy = gy_new

#
  print ( '' )
  print ( 'CVT_2D_SAMPLING' )
  print ( '  Normal end of execution.' )
  print ( '' )
  return

x=cvt_2d_sampling_H(100,7,20)

abc= 1
data=g
#print(data.shape)
