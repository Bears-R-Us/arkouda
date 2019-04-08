/* This is a primer on using the MPI FFT transforms within
   Chapel.

   Nikhil Padmanabhan
   Yale, March 2019
*/

/* Let's start by describing the problem we plan on solving.

   We'll consider a real-to-complex 3D transform and its inverse here, and will
   stick quite closely to the C API. Other transforms will follow a similar pattern,
   and we refer the reader to the FFTW documentation for more details.

   Our steps are :
   #. Declare a 3D grid, with sizes compatible to FFTW, and fill it with random numbers.
   #. Compute the sum of squares.
   #. Forward FFT, and compute the sum of squares again, showing that the data correctly transformed.
   #. Inverse transform, and compare with the original result.

*/

/* Notes on building.

   MPI programs are normally built with compiler wrappers; Chapel
   is usually build without these. If so, you will need to pass in the
   appropriate MPI flags to Chapel.

   With mpich, you can get these with 
   ```
   mpicc -compile_info -link_info
   ```
   or
   ```
   mpicc -show
   ```
   On my system, these convert to the following Chapel flags
   ```
   export CHPL_MPI_FLAGS="--ldflags='-Wl,-Bsymbolic-functions' --ldflags='-Wl,-z,relro' -I/usr/include/mpich -L/usr/lib/x86_64-linux-gnu -lmpich"
   ```

   I assume that FFTW is in the appropriate paths.

   To compile the code :
   ```
   chpl -O --fast $CHPL_MPI_FLAGS fftw-mpi.chpl \
      -lfftw3_mpi -lfftw3_threads -lfftw3 \
      --ccflags="-Wno-incompatible-pointer-types"
   ```
   The second line just lists all the required FFT libraries, while
   the third turns off warnings about incompatible pointer types.
*/


// Modules we need. Note that we use the existing
// Chapel FFTW module for some common routines. We also
// require MPI here. 
use MPI;
use FFTW;
use SysCTypes;
require "fftw3-mpi.h";

// Allow the user to define the number of threads used by FFTW.
config const numFFTWThreads=0;


// Initialize the FFTW library
// We have to do this on every locale, since
// locales correspond to MPI ranks.
coforall loc in Locales {
  on loc {

    fftw_init_threads();
    fftw_mpi_init();

    const nth = if numFFTWThreads > 0 then numFFTWThreads : c_int
      else here.maxTaskPar : c_int;
    fftw_plan_with_nthreads(nth);
  }
}


/* Now that we have everything initialized, we can
   start setting up the necessary arrays.

   FFTW assumes a slab-decomposition of the array, which for 3D grids
   implies that we distribute along the first dimension of the array.
   We use the Block distribution to distribute the array.

   For this example, let us assume that the grid dimension is `{Ng, Ng, Ng}`.

   FFTW suggests using the `fftw_mpi_local_size*` functions to figure
   out how much memory is necessary for FFTW. However, if `Ng` is divisible
   by the number of locales, then we don't need to do anything special, so we
   use that here.
*/

use BlockDist;

// The grid size
config const Ng=256;
assert((Ng%numLocales)==0,"numLocales should divide Ng");

// Define the domain for the grid. Note that the Ng+2 on the last
// dimension is how FFTW handles the real to complex transforms and how
// it packs in the complex data.
const Space = {0.. #Ng, 0.. #Ng, 0.. #(Ng+2)};
// Set up the slab decomposition of the array
const targetLocales = reshape(Locales, {0.. #numLocales, 0..0, 0..0});
const D : domain(3) dmapped Block(boundingBox=Space,
                                  targetLocales=targetLocales)=Space;

// Define the arrays
var A, B : [D] real;


// Let's print out the distribution
writef("Running an FFTW MPI example on %i locales.\n",numLocales);
for loc in Locales {
  on loc {
    writef("Locale %i has a local subdomain : %t \n",here.id, D.localSubdomain());
  }
}


// Define the domain for the x-space array
// This just skips the additional padding at the end.
const Dx = D[..,..,0.. #Ng];

// Now fill in the array, and save the result
use Random;
fillRandom(A[Dx]);
B = A;
writef("Array initialization is now complete.\n");


// Save the sum and the sum of squares
// We could do this with a reduce statement, but this
// demonstrates how to iterate over all the elements in
// x space.
var sum, sum2 : real;
forall ix in Dx with (+ reduce sum,
                      + reduce sum2) {
  sum += A[ix];
  sum2+= A[ix]**2;
}
writef("Sum, sum of squares completed.\n");

/*
 Now construct the plan for the forward transform.

 CHPL_COMM_WORLD is a custom MPI communicator, usually just
 equal to MPI_COMM_WORLD, which attempts to ensure that rank and
 locale ids are the same.

 We also protect the FFTW MPI calls to ensure that we have completed
 all Chapel communication; this prevents deadlocks in the code. Note
 that we use the MPI module provided `Barrier` to implement a
 non-blocking barrier.

 An important subtlety here - we need to call this on every locale and the
 plan must exist on every locale. In this code, we do this by just
 doing everything in an `on` block. If we were reusing the plan, we'd
 save everything in eg. a ReplicatedVar, or a PrivateDist.

*/
coforall loc in Locales {
  on loc {
    Barrier(CHPL_COMM_WORLD);

    // Get a pointer to the local part of the array
    const localIndex = (A.localSubdomain()).first;
    var Aptr = c_ptrTo(A.localAccess[localIndex]);

    // This plan is local to the locale
    var r2c_plan = fftw_mpi_plan_dft_r2c_3d(Ng : c_ptrdiff,
                                            Ng : c_ptrdiff,
                                            Ng : c_ptrdiff,
                                            Aptr, Aptr,
                                            CHPL_COMM_WORLD, FFTW_ESTIMATE);
    Barrier(CHPL_COMM_WORLD);
    // Run the FFT
    execute(r2c_plan);
    Barrier(CHPL_COMM_WORLD);
    // Clean up
    destroy_plan(r2c_plan);
    Barrier(CHPL_COMM_WORLD);
  }
}

// The zeroth frequency is just the sum of all elements
writef("Sum in x space : %er\n", sum);
writef("Sum in k space : %er\n", A[0,0,0]);
writef("Difference in sums (should be zero) : %er\n", sum-A[0,0,0]);

/* For a real-to-complex transform done in place, the array
   is interpreted as a {Ng,Ng,Ng/2+1} array of complex
   numbers (hence the extra +2 in the last dimension, with
   the negatives of the frequencies not stored being determined
   by the reality condition.

   We define a few subdomains to iterate over the k values,
   plus the real and imaginary parts.
*/

// The frequencies. Note that because of the decomposition
// along the first axis, this is distributed in the same way. 
const Dk = D[..,..,0.. #(Ng/2 + 1)];

// Real and imaginary parts
const Dre = D[..,..,.. by 2 align 0];
const Dim = D[..,..,.. by 2 align 1];

// Test the sum of squares. These should be the same (after normalizing by N^3)
// by Parseval's theorem, but we need to correctly handle the negative
// frequencies.
//
// Note that if kz=0, then the negative frequencies are stored on
// the grid.
var ksum2 : real;
forall (ik, ire, iim) in zip(Dk, Dre, Dim) with (+ reduce ksum2) {
  const fac = if ((ik(3) ==0)||(ik(3)==(Ng/2))) then 1 else 2;
  ksum2 += fac*(A[ire]**2 + A[iim]**2);
}
ksum2 /= (Ng:real)**3;

// Compare results
writef("Sum of squares in x space : %er\n", sum2);
writef("Sum of squares in k space : %er\n", ksum2);
writef("Difference in sums (should be zero) : %er\n", sum2-ksum2);

// Setup for the inverse FFT and do it.
// This follows the same structure as the forward FFT
// transpose.
coforall loc in Locales {
  on loc {
    Barrier(CHPL_COMM_WORLD);

    // Get a pointer to the local part of the array
    const localIndex = (A.localSubdomain()).first;
    var Aptr = c_ptrTo(A.localAccess[localIndex]);

    // This plan is local to the locale
    var c2r_plan = fftw_mpi_plan_dft_c2r_3d(Ng : c_ptrdiff,
                                            Ng : c_ptrdiff,
                                            Ng : c_ptrdiff,
                                            Aptr, Aptr,
                                            CHPL_COMM_WORLD, FFTW_ESTIMATE);
    Barrier(CHPL_COMM_WORLD);
    // Run the FFT
    execute(c2r_plan);
    Barrier(CHPL_COMM_WORLD);
    // Clean up
    destroy_plan(c2r_plan);
    Barrier(CHPL_COMM_WORLD);
  }
}

// The inverse FFT is unnormalized
A /= (Ng:real)**3;

// Check that the inverse worked, by comparing
// to the original array.
var diff = max reduce abs(A[Dx]-B[Dx]);
writef("Max difference between A and B : %er \n",diff);

// Here is where we clean up.
// Again, we must run this on all locales.
coforall loc in Locales {
  on loc {
    fftw_mpi_cleanup();
    fftw_cleanup_threads();
  }
}


/* The End */
writef("Brad, I hope you enjoyed this demonstration\n");


// These are the external C declarations
extern const FFTW_MPI_TRANSPOSED_IN : c_uint;
extern const FFTW_MPI_TRANSPOSED_OUT : c_uint;
extern proc fftw_set_timelimit(seconds : c_double);
extern proc fftw_init_threads(): int;
extern proc fftw_cleanup_threads();
extern proc fftw_plan_with_nthreads(nthreads: c_int);
extern proc fftw_mpi_init();
extern proc fftw_mpi_cleanup();
extern proc fftw_mpi_plan_dft_r2c_3d(n0 : c_ptrdiff, n1 : c_ptrdiff, n2 : c_ptrdiff,
                                     inarr : c_ptr(c_double) , outarr : c_ptr(c_double),
                                     comm : MPI_Comm, flags : c_uint) : fftw_plan;
extern proc fftw_mpi_plan_dft_c2r_3d(n0 : c_ptrdiff, n1 : c_ptrdiff, n2 : c_ptrdiff,
                                     inarr : c_ptr(c_double) , outarr : c_ptr(c_double),
                                     comm : MPI_Comm, flags : c_uint) : fftw_plan;
