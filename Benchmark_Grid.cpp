/*
 * Copyright © 2015 Peter Boyle <paboyle@ph.ed.ac.uk>
 * Copyright © 2024 Simon Buerger <simon.buerger@rwth-aachen.de>
 * Copyright © 2022-2025 Antonin Portelli <antonin.portelli@me.com>
 *
 * This is a fork of Benchmark_ITT.cpp from Grid
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Common.hpp"
#include "json.hpp"
#include <Grid/Grid.h>

using namespace Grid;

int NN_global;

nlohmann::json json_results;

// NOTE: Grid::GridClock is just a typedef to
// `std::chrono::high_resolution_clock`, but `Grid::usecond` rounds to
// microseconds (no idea why, probably wasnt ever relevant before), so we need
// our own wrapper here.
double usecond_precise()
{
  using namespace std::chrono;
  auto nsecs = duration_cast<nanoseconds>(GridClock::now() - Grid::theProgramStart);
  return nsecs.count() * 1e-3;
}

std::vector<std::string> get_mpi_hostnames()
{
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  char hostname[MPI_MAX_PROCESSOR_NAME];
  int name_len = 0;
  MPI_Get_processor_name(hostname, &name_len);

  // Allocate buffer to gather all hostnames
  std::vector<char> all_hostnames(world_size * MPI_MAX_PROCESSOR_NAME);

  // Use MPI_Allgather to gather all hostnames on all ranks
  MPI_Allgather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_hostnames.data(),
                MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);

  // Convert the gathered hostnames back into a vector of std::string
  std::vector<std::string> hostname_list(world_size);
  for (int i = 0; i < world_size; ++i)
  {
    hostname_list[i] = std::string(&all_hostnames[i * MPI_MAX_PROCESSOR_NAME]);
  }

  return hostname_list;
}

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
    Grid::Coordinate shm(4, 1);
    GlobalSharedMemory::GetShmDims(mpi, shm);

    uint64_t NP = TmpGrid->RankCount();
    uint64_t NN = TmpGrid->NodeCount();
    NN_global = NN;
    uint64_t SHM = NP / NN;

    grid_big_sep();
    std::cout << GridLogMessage << "Grid Default Decomposition patterns\n";
    grid_small_sep();
    std::cout << GridLogMessage << "* OpenMP threads : " << GridThread::GetThreads()
              << std::endl;

    std::cout << GridLogMessage << "* MPI layout     : " << GridCmdVectorIntToString(mpi)
              << std::endl;
    std::cout << GridLogMessage << "* Shm layout     : " << GridCmdVectorIntToString(shm)
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
      tmp["shm"].push_back(shm[i]);
    }
    tmp["ranks"] = NP;
    tmp["nodes"] = NN;
    json_results["geometry"] = tmp;
  }

  static void Comms(void)
  {
    const int Nwarmup = 50;
    int Nloop = 200;
    int nmu = 0;
    int maxlat = 48;

    Coordinate simd_layout = GridDefaultSimd(Nd, vComplexD::Nsimd());
    Coordinate mpi_layout = GridDefaultMpi();
    Coordinate shm_layout(Nd, 1);
    GlobalSharedMemory::GetShmDims(mpi_layout, shm_layout);

    for (int mu = 0; mu < Nd; mu++)
      if (mpi_layout[mu] > 1)
        nmu++;

    std::vector<double> t_time(Nloop);
    time_statistics timestat;

    std::cout << GridLogMessage << "Benchmarking threaded STENCIL halo exchange in "
              << nmu << " dimensions" << std::endl;
    grid_small_sep();
    grid_printf("%5s %5s %7s %15s %15s %15s %15s %15s\n", "L", "dir", "shm",
                "payload (B)", "time (usec)", "rate (GB/s/node)", "std dev", "max");

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
        if (mpi_layout[mu] == 1) // skip directions that are not distributed
          continue;
        bool is_shm = mpi_layout[mu] == shm_layout[mu];
        bool is_partial_shm = !is_shm && shm_layout[mu] != 1;

        std::vector<double> times(Nloop);
        for (int i = 0; i < Nwarmup; i++)
        {
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
          Grid.SendToRecvFrom((void *)&xbuf[dir][0], xmit_to_rank, (void *)&rbuf[dir][0],
                              recv_from_rank, bytes);
        }
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
          Grid.SendToRecvFrom((void *)&xbuf[dir][0], xmit_to_rank, (void *)&rbuf[dir][0],
                              recv_from_rank, bytes);
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
        grid_printf("%5d %5d %7s %15llu %15.2f %15.2f %15.1f %15.2f\n", lat, dir,
                    is_shm           ? "yes"
                    : is_partial_shm ? "partial"
                                     : "no",
                    bytes, timestat.mean, rate, rate_err, rate_max);
        nlohmann::json tmp;
        nlohmann::json tmp_rate;
        tmp["L"] = lat;
        tmp["dir"] = dir;
        tmp["shared_mem"] = is_shm;
        tmp["partial_shared_mem"] = is_partial_shm;
        tmp["bytes"] = bytes;
        tmp["time_usec"] = timestat.mean;
        tmp_rate["mean"] = rate;
        tmp_rate["error"] = rate_err;
        tmp_rate["max"] = rate_max;
        tmp["rate_GBps"] = tmp_rate;
        json_results["comms"].push_back(tmp);
      }
      for (int d = 0; d < 8; d++)
      {
        acceleratorFreeDevice(xbuf[d]);
        acceleratorFreeDevice(rbuf[d]);
      }
    }
    return;
  }

  static void Latency(void)
  {
    int Nwarmup = 100;
    int Nloop = 300;

    std::cout << GridLogMessage << "Benchmarking point-to-point latency" << std::endl;
    grid_small_sep();
    grid_printf("from to      mean(usec)           err           max\n");

    int ranks;
    int me;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    int bytes = 8;
    void *buf_from = acceleratorAllocDevice(bytes);
    void *buf_to = acceleratorAllocDevice(bytes);
    nlohmann::json json_latency;
    for (int from = 0; from < ranks; ++from)
      for (int to = 0; to < ranks; ++to)
      {
        if (from == to)
          continue;

        std::vector<double> t_time(Nloop);
        time_statistics timestat;
        MPI_Status status;

        for (int i = -Nwarmup; i < Nloop; ++i)
        {
          double start = usecond_precise();
          if (from == me)
          {
            auto err = MPI_Send(buf_from, bytes, MPI_CHAR, to, 0, MPI_COMM_WORLD);
            assert(err == MPI_SUCCESS);
          }
          if (to == me)
          {
            auto err =
                MPI_Recv(buf_to, bytes, MPI_CHAR, from, 0, MPI_COMM_WORLD, &status);
            assert(err == MPI_SUCCESS);
          }
          double stop = usecond_precise();
          if (i >= 0)
            t_time[i] = stop - start;
        }
        // important: only 'from' and 'to' have meaningful timings. we use
        // 'from's.
        MPI_Bcast(t_time.data(), Nloop, MPI_DOUBLE, from, MPI_COMM_WORLD);

        timestat.statistics(t_time);
        grid_printf("%2d %2d %15.4f %15.3f %15.4f\n", from, to, timestat.mean,
                    timestat.err, timestat.max);
        nlohmann::json tmp;
        tmp["from"] = from;
        tmp["to"] = to;
        tmp["time_usec"] = timestat.mean;
        tmp["time_usec_error"] = timestat.err;
        tmp["time_usec_min"] = timestat.min;
        tmp["time_usec_max"] = timestat.max;
        tmp["time_usec_full"] = t_time;
        json_latency.push_back(tmp);
      }
    json_results["latency"] = json_latency;

    acceleratorFreeDevice(buf_from);
    acceleratorFreeDevice(buf_to);
  }

  static void P2P(void)
  {
    // IMPORTANT: The P2P benchmark uses "MPI_COMM_WORLD" communicator, which is
    // not the quite the same as Grid.communicator. Practically speaking, the
    // latter one contains the same MPI-ranks but in a different order. Grid
    // does this make sure it can exploit ranks with shared memory (i.e.
    // multiple ranks on the same node) as best as possible.

    // buffer-size to benchmark. This number is the same as the largest one used
    // in the "Comms()" benchmark. ( L=48, Ls=12, double-prec-complex,
    // half-color-spin-vector. ). Mostly an arbitrary choice, but nice to match
    // it here
    size_t bytes = 127401984;

    int Nwarmup = 20;
    int Nloop = 100;

    std::cout << GridLogMessage << "Benchmarking point-to-point bandwidth" << std::endl;
    grid_small_sep();
    grid_printf("from to      mean(usec)           err           min           "
                "bytes    rate (GiB/s)\n");

    int ranks;
    int me;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    void *buf_from = acceleratorAllocDevice(bytes);
    void *buf_to = acceleratorAllocDevice(bytes);
    nlohmann::json json_p2p;
    for (int from = 0; from < ranks; ++from)
      for (int to = 0; to < ranks; ++to)
      {
        if (from == to)
          continue;

        std::vector<double> t_time(Nloop);
        time_statistics timestat;
        MPI_Status status;

        for (int i = -Nwarmup; i < Nloop; ++i)
        {
          double start = usecond_precise();
          if (from == me)
          {
            auto err = MPI_Send(buf_from, bytes, MPI_CHAR, to, 0, MPI_COMM_WORLD);
            assert(err == MPI_SUCCESS);
          }
          if (to == me)
          {
            auto err =
                MPI_Recv(buf_to, bytes, MPI_CHAR, from, 0, MPI_COMM_WORLD, &status);
            assert(err == MPI_SUCCESS);
          }
          double stop = usecond_precise();
          if (i >= 0)
            t_time[i] = stop - start;
        }
        // important: only 'from' and 'to' have meaningful timings. we use
        // 'from's.
        MPI_Bcast(t_time.data(), Nloop, MPI_DOUBLE, from, MPI_COMM_WORLD);

        timestat.statistics(t_time);
        double rate = bytes / (timestat.mean / 1.e6) / 1024. / 1024. / 1024.;
        double rate_err = rate * timestat.err / timestat.mean;
        double rate_max = rate * timestat.mean / timestat.min;
        double rate_min = rate * timestat.mean / timestat.max;

        grid_printf("%2d %2d %15.4f %15.3f %15.4f %15zu %15.2f\n", from, to,
                    timestat.mean, timestat.err, timestat.min, bytes, rate);

        nlohmann::json tmp;
        tmp["from"] = from;
        tmp["to"] = to;
        tmp["bytes"] = bytes;
        tmp["time_usec"] = timestat.mean;
        tmp["time_usec_error"] = timestat.err;
        tmp["time_usec_min"] = timestat.min;
        tmp["time_usec_max"] = timestat.max;
        tmp["time_usec_full"] = t_time;
        nlohmann::json tmp_rate;
        tmp_rate["mean"] = rate;
        tmp_rate["error"] = rate_err;
        tmp_rate["max"] = rate_max;
        tmp_rate["min"] = rate_min;
        tmp["rate_GBps"] = tmp_rate;

        json_p2p.push_back(tmp);
      }
    json_results["p2p"] = json_p2p;

    acceleratorFreeDevice(buf_from);
    acceleratorFreeDevice(buf_to);
  }

  static void Memory(void)
  {
    const int Nwarmup = 50;
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

    GridSerialRNG sRNG;
    sRNG.SeedFixedIntegers(std::vector<int>({45, 12, 81, 9}));
    for (int lat = 8; lat <= lmax; lat += 8)
    {
      Coordinate latt_size({lat * mpi_layout[0], lat * mpi_layout[1], lat * mpi_layout[2],
                            lat * mpi_layout[3]});
      double vol =
          static_cast<double>(latt_size[0]) * latt_size[1] * latt_size[2] * latt_size[3];

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

      const uint64_t Nloop = (200 * lmax * lmax * lmax / lat / lat / lat);

      for (int i = 0; i < Nwarmup; i++)
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
    const int Nwarmup = 50;
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
      double vol =
          static_cast<double>(latt_size[0]) * latt_size[1] * latt_size[2] * latt_size[3];

      GridCartesian Grid(latt_size, simd_layout, mpi_layout);

      NN = Grid.NodeCount();

      LatticeSU4 z(&Grid);
      z = Zero();
      LatticeSU4 x(&Grid);
      x = Zero();
      LatticeSU4 y(&Grid);
      y = Zero();

      const uint64_t Nloop = (200 * lmax * lmax * lmax / lat / lat / lat);

      for (int i = 0; i < Nwarmup; i++)
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

  int Ls = 1;
  bool do_su4 = true;
  bool do_memory = true;
  bool do_comms = true;
  bool do_flops = true;

  // NOTE: these two take O((number of ranks)^2) time, which might be a lot, so they are
  // off by default
  bool do_latency = false;
  bool do_p2p = false;

  std::string json_filename = ""; // empty indicates no json output
  for (int i = 0; i < argc; i++)
  {
    auto arg = std::string(argv[i]);
    if (arg == "--json-out")
      json_filename = argv[i + 1];
    if (arg == "--benchmark-su4")
      do_su4 = true;
    if (arg == "--benchmark-memory")
      do_memory = true;
    if (arg == "--benchmark-comms")
      do_comms = true;
    if (arg == "--benchmark-flops")
      do_flops = true;
    if (arg == "--benchmark-latency")
      do_latency = true;
    if (arg == "--benchmark-p2p")
      do_p2p = true;
    if (arg == "--no-benchmark-su4")
      do_su4 = false;
    if (arg == "--no-benchmark-memory")
      do_memory = false;
    if (arg == "--no-benchmark-comms")
      do_comms = false;
    if (arg == "--no-benchmark-flops")
      do_flops = false;
    if (arg == "--no-benchmark-latency")
      do_latency = false;
    if (arg == "--no-benchmark-p2p")
      do_p2p = false;
  }

  CartesianCommunicator::SetCommunicatorPolicy(
      CartesianCommunicator::CommunicatorPolicySequential);

  Benchmark::Decomposition();

  int sel = 4;
  std::vector<int> L_list({8, 12, 16, 24, 32, 48});
  int selm1 = sel - 1;

  std::vector<double> wilson;
  std::vector<double> dwf4;
  std::vector<double> staggered;

  auto runBenchmark = [](const std::string& name, const std::function<void(void)>& fn)
  {
    grid_big_sep();
    std::cout << GridLogMessage << " " << name << " benchmark " << std::endl;
    grid_big_sep();
    fn();
  };

  if (do_memory)  runBenchmark("Memory",         &Benchmark::Memory);
  if (do_su4)     runBenchmark("SU(4)",          &Benchmark::SU4);
  if (do_comms)   runBenchmark("Communications", &Benchmark::Comms);
  if (do_latency) runBenchmark("Latency",        &Benchmark::Latency);
  if (do_p2p)     runBenchmark("Point-To-Point", &Benchmark::P2P);

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

  json_results["hostnames"] = get_mpi_hostnames();

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
