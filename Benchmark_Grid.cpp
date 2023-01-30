/*
Copyright © 2015 Peter Boyle <paboyle@ph.ed.ac.uk>
Copyright © 2022 Antonin Portelli <antonin.portelli@me.com>
Copyright © 2022 Simon Buerger <simon.buerger@rwth-aachen.de>

This is a fork of Benchmark_ITT.cpp from Grid

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Common.hpp"
#include "json.hpp"
#include <Grid/Grid.h>

using namespace Grid;

int NN_global;

nlohmann::json json_results;

struct time_statistics
{
  double mean;
  double err;
  double min;
  double max;

  void statistics(std::vector<double> v)
  {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    mean = sum / v.size();

    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [=](double x) { return x - mean; });
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    err = std::sqrt(sq_sum / (v.size() * (v.size() - 1)));

    auto result = std::minmax_element(v.begin(), v.end());
    min = *result.first;
    max = *result.second;
  }
};

struct controls
{
  int Opt;
  int CommsOverlap;
  Grid::CartesianCommunicator::CommunicatorPolicy_t CommsAsynch;
};

class Benchmark
{
  public:
  static void Decomposition(void)
  {
    nlohmann::json tmp;
    int threads = GridThread::GetThreads();
    Grid::Coordinate mpi = GridDefaultMpi();
    assert(mpi.size() == 4);
    Coordinate local({8, 8, 8, 8});
    Coordinate latt4(
        {local[0] * mpi[0], local[1] * mpi[1], local[2] * mpi[2], local[3] * mpi[3]});
    GridCartesian *TmpGrid = SpaceTimeGrid::makeFourDimGrid(
        latt4, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());

    uint64_t NP = TmpGrid->RankCount();
    uint64_t NN = TmpGrid->NodeCount();
    NN_global = NN;
    uint64_t SHM = NP / NN;

    grid_big_sep();
    std::cout << GridLogMessage << "Grid Default Decomposition patterns\n";
    grid_small_sep();
    std::cout << GridLogMessage << "* OpenMP threads : " << GridThread::GetThreads()
              << std::endl;

    std::cout << GridLogMessage << "* MPI tasks      : " << GridCmdVectorIntToString(mpi)
              << std::endl;

    std::cout << GridLogMessage << "* vReal          : " << sizeof(vReal) * 8 << "bits ; "
              << GridCmdVectorIntToString(GridDefaultSimd(4, vReal::Nsimd()))
              << std::endl;
    std::cout << GridLogMessage << "* vRealF         : " << sizeof(vRealF) * 8
              << "bits ; "
              << GridCmdVectorIntToString(GridDefaultSimd(4, vRealF::Nsimd()))
              << std::endl;
    std::cout << GridLogMessage << "* vRealD         : " << sizeof(vRealD) * 8
              << "bits ; "
              << GridCmdVectorIntToString(GridDefaultSimd(4, vRealD::Nsimd()))
              << std::endl;
    std::cout << GridLogMessage << "* vComplex       : " << sizeof(vComplex) * 8
              << "bits ; "
              << GridCmdVectorIntToString(GridDefaultSimd(4, vComplex::Nsimd()))
              << std::endl;
    std::cout << GridLogMessage << "* vComplexF      : " << sizeof(vComplexF) * 8
              << "bits ; "
              << GridCmdVectorIntToString(GridDefaultSimd(4, vComplexF::Nsimd()))
              << std::endl;
    std::cout << GridLogMessage << "* vComplexD      : " << sizeof(vComplexD) * 8
              << "bits ; "
              << GridCmdVectorIntToString(GridDefaultSimd(4, vComplexD::Nsimd()))
              << std::endl;
    std::cout << GridLogMessage << "* ranks          : " << NP << std::endl;
    std::cout << GridLogMessage << "* nodes          : " << NN << std::endl;
    std::cout << GridLogMessage << "* ranks/node     : " << SHM << std::endl;

    for (unsigned int i = 0; i < mpi.size(); ++i)
    {
      tmp["mpi"].push_back(mpi[i]);
    }
    tmp["ranks"] = NP;
    tmp["nodes"] = NN;
    json_results["geometry"] = tmp;
  }

  static void Comms(void)
  {
    int Nloop = 200;
    int nmu = 0;
    int maxlat = 48;

    Coordinate simd_layout = GridDefaultSimd(Nd, vComplexD::Nsimd());
    Coordinate mpi_layout = GridDefaultMpi();

    for (int mu = 0; mu < Nd; mu++)
      if (mpi_layout[mu] > 1)
        nmu++;

    std::vector<double> t_time(Nloop);
    time_statistics timestat;

    std::cout << GridLogMessage << "Benchmarking threaded STENCIL halo exchange in "
              << nmu << " dimensions" << std::endl;
    grid_small_sep();
    grid_printf("%5s %5s %15s %15s %15s %15s %15s\n", "L", "dir", "payload (B)",
                "time (usec)", "rate (GB/s)", "std dev", "max");

    for (int lat = 16; lat <= maxlat; lat += 8)
    {
      int Ls = 12;

      Coordinate latt_size({lat * mpi_layout[0], lat * mpi_layout[1], lat * mpi_layout[2],
                            lat * mpi_layout[3]});

      GridCartesian Grid(latt_size, simd_layout, mpi_layout);
      RealD Nrank = Grid._Nprocessors;
      RealD Nnode = Grid.NodeCount();
      RealD ppn = Nrank / Nnode;

      std::vector<HalfSpinColourVectorD *> xbuf(8);
      std::vector<HalfSpinColourVectorD *> rbuf(8);
      uint64_t bytes = lat * lat * lat * Ls * sizeof(HalfSpinColourVectorD);
      for (int d = 0; d < 8; d++)
      {
        xbuf[d] = (HalfSpinColourVectorD *)acceleratorAllocDevice(bytes);
        rbuf[d] = (HalfSpinColourVectorD *)acceleratorAllocDevice(bytes);
      }

      double dbytes;

      for (int dir = 0; dir < 8; dir++)
      {
        int mu = dir % 4;
        if (mpi_layout[mu] > 1)
        {

          std::vector<double> times(Nloop);
          for (int i = 0; i < Nloop; i++)
          {

            dbytes = 0;
            double start = usecond();
            int xmit_to_rank;
            int recv_from_rank;

            if (dir == mu)
            {
              int comm_proc = 1;
              Grid.ShiftedRanks(mu, comm_proc, xmit_to_rank, recv_from_rank);
            }
            else
            {
              int comm_proc = mpi_layout[mu] - 1;
              Grid.ShiftedRanks(mu, comm_proc, xmit_to_rank, recv_from_rank);
            }
            Grid.SendToRecvFrom((void *)&xbuf[dir][0], xmit_to_rank,
                                (void *)&rbuf[dir][0], recv_from_rank, bytes);
            dbytes += bytes;

            double stop = usecond();
            t_time[i] = stop - start; // microseconds
          }
          timestat.statistics(t_time);

          dbytes = dbytes * ppn;
          double bidibytes = 2. * dbytes;
          double rate = bidibytes / (timestat.mean / 1.e6) / 1024. / 1024. / 1024.;
          double rate_err = rate * timestat.err / timestat.mean;
          double rate_max = rate * timestat.mean / timestat.min;
          grid_printf("%5d %5d %15d %15.2f %15.2f %15.1f %15.2f\n", lat, dir, bytes,
                      timestat.mean, rate, rate_err, rate_max);
          nlohmann::json tmp;
          nlohmann::json tmp_rate;
          tmp["L"] = lat;
          tmp["dir"] = dir;
          tmp["bytes"] = bytes;
          tmp["time_usec"] = timestat.mean;
          tmp_rate["mean"] = rate;
          tmp_rate["error"] = rate_err;
          tmp_rate["max"] = rate_max;
          tmp["rate_GBps"] = tmp_rate;
          json_results["comms"].push_back(tmp);
        }
      }
      for (int d = 0; d < 8; d++)
      {
        acceleratorFreeDevice(xbuf[d]);
        acceleratorFreeDevice(rbuf[d]);
      }
    }
    return;
  }

  static void Memory(void)
  {
    const int Nvec = 8;
    typedef Lattice<iVector<vReal, Nvec>> LatticeVec;
    typedef iVector<vReal, Nvec> Vec;

    Coordinate simd_layout = GridDefaultSimd(Nd, vReal::Nsimd());
    Coordinate mpi_layout = GridDefaultMpi();

    std::cout << GridLogMessage << "Benchmarking a*x + y bandwidth" << std::endl;
    grid_small_sep();
    grid_printf("%5s %15s %15s %15s %15s\n", "L", "size (MB/node)", "time (usec)",
                "GB/s/node", "Gflop/s/node");

    uint64_t NN;
    uint64_t lmax = 64;
#define NLOOP (200 * lmax * lmax * lmax / lat / lat / lat)
#define NWARMUP 50

    GridSerialRNG sRNG;
    sRNG.SeedFixedIntegers(std::vector<int>({45, 12, 81, 9}));
    for (int lat = 8; lat <= lmax; lat += 8)
    {

      Coordinate latt_size({lat * mpi_layout[0], lat * mpi_layout[1], lat * mpi_layout[2],
                            lat * mpi_layout[3]});
      uint64_t vol = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3];

      GridCartesian Grid(latt_size, simd_layout, mpi_layout);

      NN = Grid.NodeCount();

      Vec rn;
      random(sRNG, rn);

      LatticeVec z(&Grid);
      z = Zero();
      LatticeVec x(&Grid);
      x = Zero();
      LatticeVec y(&Grid);
      y = Zero();
      double a = 2.0;

      uint64_t Nloop = NLOOP;

      for (int i = 0; i < NWARMUP; i++)
      {
        z = a * x - y;
      }
      double start = usecond();
      for (int i = 0; i < Nloop; i++)
      {
        z = a * x - y;
      }
      double stop = usecond();
      double time = (stop - start) / Nloop / 1.e6;

      double flops = vol * Nvec * 2 / 1.e9; // mul,add
      double bytes = 3.0 * vol * Nvec * sizeof(Real) / 1024. / 1024.;

      grid_printf("%5d %15.2f %15.2f %15.2f %15.2f\n", lat, bytes / NN, time * 1.e6,
                  bytes / time / NN / 1024., flops / time / NN);

      nlohmann::json tmp;
      tmp["L"] = lat;
      tmp["size_MB"] = bytes / NN;
      tmp["GBps"] = bytes / time / NN / 1024.;
      tmp["GFlops"] = flops / time / NN;
      json_results["axpy"].push_back(tmp);
    }
  };

  static void SU4(void)
  {
    const int Nc4 = 4;
    typedef Lattice<iMatrix<vComplexF, Nc4>> LatticeSU4;

    Coordinate simd_layout = GridDefaultSimd(Nd, vComplexF::Nsimd());
    Coordinate mpi_layout = GridDefaultMpi();

    std::cout << GridLogMessage << "Benchmarking z = y*x SU(4) bandwidth" << std::endl;
    grid_small_sep();
    grid_printf("%5s %15s %15s %15s %15s\n", "L", "size (MB/node)", "time (usec)",
                "GB/s/node", "Gflop/s/node");

    uint64_t NN;

    uint64_t lmax = 48;

    GridSerialRNG sRNG;
    sRNG.SeedFixedIntegers(std::vector<int>({45, 12, 81, 9}));
    for (int lat = 8; lat <= lmax; lat += 8)
    {

      Coordinate latt_size({lat * mpi_layout[0], lat * mpi_layout[1], lat * mpi_layout[2],
                            lat * mpi_layout[3]});
      int64_t vol = latt_size[0] * latt_size[1] * latt_size[2] * latt_size[3];

      GridCartesian Grid(latt_size, simd_layout, mpi_layout);

      NN = Grid.NodeCount();

      LatticeSU4 z(&Grid);
      z = Zero();
      LatticeSU4 x(&Grid);
      x = Zero();
      LatticeSU4 y(&Grid);
      y = Zero();

      uint64_t Nloop = NLOOP;

      for (int i = 0; i < NWARMUP; i++)
      {
        z = x * y;
      }
      double start = usecond();
      for (int i = 0; i < Nloop; i++)
      {
        z = x * y;
      }
      double stop = usecond();
      double time = (stop - start) / Nloop / 1.e6;

      double flops = vol * Nc4 * Nc4 * (6 + (Nc4 - 1) * 8) / 1.e9; // mul,add
      double bytes = 3.0 * vol * Nc4 * Nc4 * 2 * sizeof(RealF) / 1024. / 1024.;
      grid_printf("%5d %15.2f %15.2f %15.2f %15.2f\n", lat, bytes / NN, time * 1.e6,
                  bytes / time / NN / 1024., flops / time / NN);

      nlohmann::json tmp;
      tmp["L"] = lat;
      tmp["size_MB"] = bytes / NN;
      tmp["GBps"] = bytes / time / NN / 1024.;
      tmp["GFlops"] = flops / time / NN;
      json_results["SU4"].push_back(tmp);
    }
  };

  static double DWF(int Ls, int L)
  {
    RealD mass = 0.1;
    RealD M5 = 1.8;

    double gflops;
    double gflops_best = 0;
    double gflops_worst = 0;
    std::vector<double> gflops_all;

    ///////////////////////////////////////////////////////
    // Set/Get the layout & grid size
    ///////////////////////////////////////////////////////
    int threads = GridThread::GetThreads();
    Coordinate mpi = GridDefaultMpi();
    assert(mpi.size() == 4);
    Coordinate local({L, L, L, L});
    Coordinate latt4(
        {local[0] * mpi[0], local[1] * mpi[1], local[2] * mpi[2], local[3] * mpi[3]});

    GridCartesian *TmpGrid = SpaceTimeGrid::makeFourDimGrid(
        latt4, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
    uint64_t NP = TmpGrid->RankCount();
    uint64_t NN = TmpGrid->NodeCount();
    NN_global = NN;
    uint64_t SHM = NP / NN;

    ///////// Welcome message ////////////
    grid_big_sep();
    std::cout << GridLogMessage << "Benchmark DWF on " << L << "^4 local volume "
              << std::endl;
    std::cout << GridLogMessage << "* Nc             : " << Nc << std::endl;
    std::cout << GridLogMessage
              << "* Global volume  : " << GridCmdVectorIntToString(latt4) << std::endl;
    std::cout << GridLogMessage << "* Ls             : " << Ls << std::endl;
    std::cout << GridLogMessage << "* ranks          : " << NP << std::endl;
    std::cout << GridLogMessage << "* nodes          : " << NN << std::endl;
    std::cout << GridLogMessage << "* ranks/node     : " << SHM << std::endl;
    std::cout << GridLogMessage << "* ranks geom     : " << GridCmdVectorIntToString(mpi)
              << std::endl;
    std::cout << GridLogMessage << "* Using " << threads << " threads" << std::endl;
    grid_big_sep();

    ///////// Lattice Init ////////////
    GridCartesian *UGrid = SpaceTimeGrid::makeFourDimGrid(
        latt4, GridDefaultSimd(Nd, vComplexF::Nsimd()), GridDefaultMpi());
    GridRedBlackCartesian *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
    GridCartesian *FGrid = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
    GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

    ///////// RNG Init ////////////
    std::vector<int> seeds4({1, 2, 3, 4});
    std::vector<int> seeds5({5, 6, 7, 8});
    GridParallelRNG RNG4(UGrid);
    RNG4.SeedFixedIntegers(seeds4);
    GridParallelRNG RNG5(FGrid);
    RNG5.SeedFixedIntegers(seeds5);
    std::cout << GridLogMessage << "Initialised RNGs" << std::endl;

    typedef DomainWallFermionF Action;
    typedef typename Action::FermionField Fermion;
    typedef LatticeGaugeFieldF Gauge;

    ///////// Source preparation ////////////
    Gauge Umu(UGrid);
    SU<Nc>::HotConfiguration(RNG4, Umu);
    Fermion src(FGrid);
    random(RNG5, src);
    Fermion src_e(FrbGrid);
    Fermion src_o(FrbGrid);
    Fermion r_e(FrbGrid);
    Fermion r_o(FrbGrid);
    Fermion r_eo(FGrid);
    Action Dw(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, M5);

    {

      pickCheckerboard(Even, src_e, src);
      pickCheckerboard(Odd, src_o, src);

      const int num_cases = 4;
      std::string fmt("G/S/C ; G/O/C ; G/S/S ; G/O/S ");

      controls Cases[] = {
          {WilsonKernelsStatic::OptGeneric, WilsonKernelsStatic::CommsThenCompute,
           CartesianCommunicator::CommunicatorPolicyConcurrent},
          {WilsonKernelsStatic::OptGeneric, WilsonKernelsStatic::CommsAndCompute,
           CartesianCommunicator::CommunicatorPolicyConcurrent},
          {WilsonKernelsStatic::OptGeneric, WilsonKernelsStatic::CommsThenCompute,
           CartesianCommunicator::CommunicatorPolicySequential},
          {WilsonKernelsStatic::OptGeneric, WilsonKernelsStatic::CommsAndCompute,
           CartesianCommunicator::CommunicatorPolicySequential}};

      for (int c = 0; c < num_cases; c++)
      {

        WilsonKernelsStatic::Comms = Cases[c].CommsOverlap;
        WilsonKernelsStatic::Opt = Cases[c].Opt;
        CartesianCommunicator::SetCommunicatorPolicy(Cases[c].CommsAsynch);

        grid_small_sep();
        if (WilsonKernelsStatic::Opt == WilsonKernelsStatic::OptGeneric)
          std::cout << GridLogMessage << "* Using GENERIC Nc WilsonKernels" << std::endl;
        if (WilsonKernelsStatic::Comms == WilsonKernelsStatic::CommsAndCompute)
          std::cout << GridLogMessage << "* Using Overlapped Comms/Compute" << std::endl;
        if (WilsonKernelsStatic::Comms == WilsonKernelsStatic::CommsThenCompute)
          std::cout << GridLogMessage << "* Using sequential Comms/Compute" << std::endl;
        std::cout << GridLogMessage << "* SINGLE precision " << std::endl;
        grid_small_sep();

        int nwarm = 10;
        double t0 = usecond();
        FGrid->Barrier();
        for (int i = 0; i < nwarm; i++)
        {
          Dw.DhopEO(src_o, r_e, DaggerNo);
        }
        FGrid->Barrier();
        double t1 = usecond();
        uint64_t ncall = 500;

        FGrid->Broadcast(0, &ncall, sizeof(ncall));

        Dw.ZeroCounters();

        time_statistics timestat;
        std::vector<double> t_time(ncall);
        for (uint64_t i = 0; i < ncall; i++)
        {
          t0 = usecond();
          Dw.DhopEO(src_o, r_e, DaggerNo);
          t1 = usecond();
          t_time[i] = t1 - t0;
        }
        FGrid->Barrier();

        double volume = Ls;
        for (int mu = 0; mu < Nd; mu++)
          volume = volume * latt4[mu];

          // Nc=3 gives
          // 1344= 3*(2*8+6)*2*8 + 8*3*2*2 + 3*4*2*8
          // 1344 = Nc* (6+(Nc-1)*8)*2*Nd + Nd*Nc*2*2  + Nd*Nc*Ns*2
          //	double flops=(1344.0*volume)/2;
#if 0
	double fps = Nc* (6+(Nc-1)*8)*Ns*Nd + Nd*Nc*Ns  + Nd*Nc*Ns*2;
#else
        double fps =
            Nc * (6 + (Nc - 1) * 8) * Ns * Nd + 2 * Nd * Nc * Ns + 2 * Nd * Nc * Ns * 2;
#endif
        double flops = (fps * volume) / 2.;
        double gf_hi, gf_lo, gf_err;

        timestat.statistics(t_time);
        gf_hi = flops / timestat.min / 1000.;
        gf_lo = flops / timestat.max / 1000.;
        gf_err = flops / timestat.min * timestat.err / timestat.mean / 1000.;

        gflops = flops / timestat.mean / 1000.;
        gflops_all.push_back(gflops);
        if (gflops_best == 0)
          gflops_best = gflops;
        if (gflops_worst == 0)
          gflops_worst = gflops;
        if (gflops > gflops_best)
          gflops_best = gflops;
        if (gflops < gflops_worst)
          gflops_worst = gflops;

        std::cout << GridLogMessage << "Deo FlopsPerSite is " << fps << std::endl;
        std::cout << GridLogMessage << std::fixed << std::setprecision(1)
                  << "Deo Gflop/s =   " << gflops << " (" << gf_err << ") " << gf_lo
                  << "-" << gf_hi << std::endl;
        std::cout << GridLogMessage << std::fixed << std::setprecision(1)
                  << "Deo Gflop/s per rank   " << gflops / NP << std::endl;
        std::cout << GridLogMessage << std::fixed << std::setprecision(1)
                  << "Deo Gflop/s per node   " << gflops / NN << std::endl;
      }

      grid_small_sep();
      std::cout << GridLogMessage << L << "^4 x " << Ls
                << " Deo Best  Gflop/s        =   " << gflops_best << " ; "
                << gflops_best / NN << " per node " << std::endl;
      std::cout << GridLogMessage << L << "^4 x " << Ls
                << " Deo Worst Gflop/s        =   " << gflops_worst << " ; "
                << gflops_worst / NN << " per node " << std::endl;
      std::cout << GridLogMessage << fmt << std::endl;
      std::cout << GridLogMessage;

      for (int i = 0; i < gflops_all.size(); i++)
      {
        std::cout << gflops_all[i] / NN << " ; ";
      }
      std::cout << std::endl;
    }
    return gflops_best;
  }

  static double Staggered(int L)
  {
    double gflops;
    double gflops_best = 0;
    double gflops_worst = 0;
    std::vector<double> gflops_all;

    ///////////////////////////////////////////////////////
    // Set/Get the layout & grid size
    ///////////////////////////////////////////////////////
    int threads = GridThread::GetThreads();
    Coordinate mpi = GridDefaultMpi();
    assert(mpi.size() == 4);
    Coordinate local({L, L, L, L});
    Coordinate latt4(
        {local[0] * mpi[0], local[1] * mpi[1], local[2] * mpi[2], local[3] * mpi[3]});

    GridCartesian *TmpGrid = SpaceTimeGrid::makeFourDimGrid(
        latt4, GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
    uint64_t NP = TmpGrid->RankCount();
    uint64_t NN = TmpGrid->NodeCount();
    NN_global = NN;
    uint64_t SHM = NP / NN;

    ///////// Welcome message ////////////
    grid_big_sep();
    std::cout << GridLogMessage << "Benchmark ImprovedStaggered on " << L
              << "^4 local volume " << std::endl;
    std::cout << GridLogMessage
              << "* Global volume  : " << GridCmdVectorIntToString(latt4) << std::endl;
    std::cout << GridLogMessage << "* ranks          : " << NP << std::endl;
    std::cout << GridLogMessage << "* nodes          : " << NN << std::endl;
    std::cout << GridLogMessage << "* ranks/node     : " << SHM << std::endl;
    std::cout << GridLogMessage << "* ranks geom     : " << GridCmdVectorIntToString(mpi)
              << std::endl;
    std::cout << GridLogMessage << "* Using " << threads << " threads" << std::endl;
    grid_big_sep();

    ///////// Lattice Init ////////////
    GridCartesian *FGrid = SpaceTimeGrid::makeFourDimGrid(
        latt4, GridDefaultSimd(Nd, vComplexF::Nsimd()), GridDefaultMpi());
    GridRedBlackCartesian *FrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(FGrid);

    ///////// RNG Init ////////////
    std::vector<int> seeds4({1, 2, 3, 4});
    GridParallelRNG RNG4(FGrid);
    RNG4.SeedFixedIntegers(seeds4);
    std::cout << GridLogMessage << "Initialised RNGs" << std::endl;

    RealD mass = 0.1;
    RealD c1 = 9.0 / 8.0;
    RealD c2 = -1.0 / 24.0;
    RealD u0 = 1.0;

    typedef ImprovedStaggeredFermionF Action;
    typedef typename Action::FermionField Fermion;
    typedef LatticeGaugeFieldF Gauge;

    Gauge Umu(FGrid);
    SU<Nc>::HotConfiguration(RNG4, Umu);

    typename Action::ImplParams params;
    Action Ds(Umu, Umu, *FGrid, *FrbGrid, mass, c1, c2, u0, params);

    ///////// Source preparation ////////////
    Fermion src(FGrid);
    random(RNG4, src);
    Fermion src_e(FrbGrid);
    Fermion src_o(FrbGrid);
    Fermion r_e(FrbGrid);
    Fermion r_o(FrbGrid);
    Fermion r_eo(FGrid);

    {

      pickCheckerboard(Even, src_e, src);
      pickCheckerboard(Odd, src_o, src);

      const int num_cases = 4;
      std::string fmt("G/S/C ; G/O/C ; G/S/S ; G/O/S ");

      controls Cases[] = {
          {StaggeredKernelsStatic::OptGeneric, StaggeredKernelsStatic::CommsThenCompute,
           CartesianCommunicator::CommunicatorPolicyConcurrent},
          {StaggeredKernelsStatic::OptGeneric, StaggeredKernelsStatic::CommsAndCompute,
           CartesianCommunicator::CommunicatorPolicyConcurrent},
          {StaggeredKernelsStatic::OptGeneric, StaggeredKernelsStatic::CommsThenCompute,
           CartesianCommunicator::CommunicatorPolicySequential},
          {StaggeredKernelsStatic::OptGeneric, StaggeredKernelsStatic::CommsAndCompute,
           CartesianCommunicator::CommunicatorPolicySequential}};

      for (int c = 0; c < num_cases; c++)
      {

        StaggeredKernelsStatic::Comms = Cases[c].CommsOverlap;
        StaggeredKernelsStatic::Opt = Cases[c].Opt;
        CartesianCommunicator::SetCommunicatorPolicy(Cases[c].CommsAsynch);

        grid_small_sep();
        if (StaggeredKernelsStatic::Opt == StaggeredKernelsStatic::OptGeneric)
          std::cout << GridLogMessage << "* Using GENERIC Nc StaggeredKernels"
                    << std::endl;
        if (StaggeredKernelsStatic::Comms == StaggeredKernelsStatic::CommsAndCompute)
          std::cout << GridLogMessage << "* Using Overlapped Comms/Compute" << std::endl;
        if (StaggeredKernelsStatic::Comms == StaggeredKernelsStatic::CommsThenCompute)
          std::cout << GridLogMessage << "* Using sequential Comms/Compute" << std::endl;
        std::cout << GridLogMessage << "* SINGLE precision " << std::endl;
        grid_small_sep();

        int nwarm = 10;
        double t0 = usecond();
        FGrid->Barrier();
        for (int i = 0; i < nwarm; i++)
        {
          Ds.DhopEO(src_o, r_e, DaggerNo);
        }
        FGrid->Barrier();
        double t1 = usecond();
        uint64_t ncall = 500;

        FGrid->Broadcast(0, &ncall, sizeof(ncall));
        Ds.ZeroCounters();

        time_statistics timestat;
        std::vector<double> t_time(ncall);
        for (uint64_t i = 0; i < ncall; i++)
        {
          t0 = usecond();
          Ds.DhopEO(src_o, r_e, DaggerNo);
          t1 = usecond();
          t_time[i] = t1 - t0;
        }
        FGrid->Barrier();

        double volume = 1;
        for (int mu = 0; mu < Nd; mu++)
          volume = volume * latt4[mu];
        double flops = (1146.0 * volume) / 2.;
        double gf_hi, gf_lo, gf_err;

        timestat.statistics(t_time);
        gf_hi = flops / timestat.min / 1000.;
        gf_lo = flops / timestat.max / 1000.;
        gf_err = flops / timestat.min * timestat.err / timestat.mean / 1000.;

        gflops = flops / timestat.mean / 1000.;
        gflops_all.push_back(gflops);
        if (gflops_best == 0)
          gflops_best = gflops;
        if (gflops_worst == 0)
          gflops_worst = gflops;
        if (gflops > gflops_best)
          gflops_best = gflops;
        if (gflops < gflops_worst)
          gflops_worst = gflops;

        std::cout << GridLogMessage << std::fixed << std::setprecision(1)
                  << "Deo Gflop/s =   " << gflops << " (" << gf_err << ") " << gf_lo
                  << "-" << gf_hi << std::endl;
        std::cout << GridLogMessage << std::fixed << std::setprecision(1)
                  << "Deo Gflop/s per rank   " << gflops / NP << std::endl;
        std::cout << GridLogMessage << std::fixed << std::setprecision(1)
                  << "Deo Gflop/s per node   " << gflops / NN << std::endl;
      }

      grid_small_sep();
      std::cout << GridLogMessage << L
                << "^4  Deo Best  Gflop/s        =   " << gflops_best << " ; "
                << gflops_best / NN << " per node " << std::endl;
      std::cout << GridLogMessage << L
                << "^4  Deo Worst Gflop/s        =   " << gflops_worst << " ; "
                << gflops_worst / NN << " per node " << std::endl;
      std::cout << GridLogMessage << fmt << std::endl;
      std::cout << GridLogMessage;

      for (int i = 0; i < gflops_all.size(); i++)
      {
        std::cout << gflops_all[i] / NN << " ; ";
      }
      std::cout << std::endl;
    }
    return gflops_best;
  }
};

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);

  std::string json_filename = ""; // empty indicates no json output
  for (int i = 0; i < argc; i++)
  {
    if (std::string(argv[i]) == "--json-out")
      json_filename = argv[i + 1];
  }

  CartesianCommunicator::SetCommunicatorPolicy(
      CartesianCommunicator::CommunicatorPolicySequential);
#ifdef KNL
  LebesgueOrder::Block = std::vector<int>({8, 2, 2, 2});
#else
  LebesgueOrder::Block = std::vector<int>({2, 2, 2, 2});
#endif
  Benchmark::Decomposition();

  int do_su4 = 1;
  int do_memory = 1;
  int do_comms = 1;
  int do_flops = 1;
  int Ls = 1;

  int sel = 4;
  std::vector<int> L_list({8, 12, 16, 24, 32});
  int selm1 = sel - 1;

  std::vector<double> wilson;
  std::vector<double> dwf4;
  std::vector<double> staggered;

  if (do_memory)
  {
    grid_big_sep();
    std::cout << GridLogMessage << " Memory benchmark " << std::endl;
    grid_big_sep();
    Benchmark::Memory();
  }

  if (do_su4)
  {
    grid_big_sep();
    std::cout << GridLogMessage << " SU(4) benchmark " << std::endl;
    grid_big_sep();
    Benchmark::SU4();
  }

  if (do_comms)
  {
    grid_big_sep();
    std::cout << GridLogMessage << " Communications benchmark " << std::endl;
    grid_big_sep();
    Benchmark::Comms();
  }

  if (do_flops)
  {
    Ls = 1;
    grid_big_sep();
    std::cout << GridLogMessage << " Wilson dslash 4D vectorised" << std::endl;
    for (int l = 0; l < L_list.size(); l++)
    {
      wilson.push_back(Benchmark::DWF(Ls, L_list[l]));
    }

    Ls = 12;
    grid_big_sep();
    std::cout << GridLogMessage << " Domain wall dslash 4D vectorised" << std::endl;
    for (int l = 0; l < L_list.size(); l++)
    {
      double result = Benchmark::DWF(Ls, L_list[l]);
      dwf4.push_back(result);
    }

    grid_big_sep();
    std::cout << GridLogMessage << " Improved Staggered dslash 4D vectorised"
              << std::endl;
    for (int l = 0; l < L_list.size(); l++)
    {
      double result = Benchmark::Staggered(L_list[l]);
      staggered.push_back(result);
    }

    int NN = NN_global;

    grid_big_sep();
    std::cout << GridLogMessage << "Gflop/s/node Summary table Ls=" << Ls << std::endl;
    grid_big_sep();
    grid_printf("%5s %12s %12s %12s\n", "L", "Wilson", "DWF", "Staggered");
    nlohmann::json tmp_flops;
    for (int l = 0; l < L_list.size(); l++)
    {
      grid_printf("%5d %12.2f %12.2f %12.2f\n", L_list[l], wilson[l] / NN, dwf4[l] / NN,
                  staggered[l] / NN);

      nlohmann::json tmp;
      tmp["L"] = L_list[l];
      tmp["Gflops_wilson"] = wilson[l] / NN;
      tmp["Gflops_dwf4"] = dwf4[l] / NN;
      tmp["Gflops_staggered"] = staggered[l] / NN;
      tmp_flops["results"].push_back(tmp);
    }
    grid_big_sep();
    std::cout << GridLogMessage
              << " Comparison point     result: " << 0.5 * (dwf4[sel] + dwf4[selm1]) / NN
              << " Gflop/s per node" << std::endl;
    std::cout << GridLogMessage << " Comparison point is 0.5*(" << dwf4[sel] / NN << "+"
              << dwf4[selm1] / NN << ") " << std::endl;
    std::cout << std::setprecision(3);
    grid_big_sep();
    tmp_flops["comparison_point_Gflops"] = 0.5 * (dwf4[sel] + dwf4[selm1]) / NN;
    json_results["flops"] = tmp_flops;
  }

  if (!json_filename.empty())
  {
    std::cout << GridLogMessage << "writing benchmark results to " << json_filename
              << std::endl;

    int me = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    if (me == 0)
    {
      std::ofstream json_file(json_filename);
      json_file << std::setw(2) << json_results;
    }
  }

  Grid_finalize();
}
