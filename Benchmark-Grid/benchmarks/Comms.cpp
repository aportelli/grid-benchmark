#include "Benchmarks.hpp"

#include <Grid/Grid.h>
#include "Common.hpp"
#include "Benchmark-Grid/Utils.hpp"

using namespace Grid;


void Benchmark::Comms(nlohmann::json& json_results)
{
  const int Nwarmup = 50;
  int Nloop = 200;
  int nmu = 0;
  int maxlat = 48;
  int Ls = 12;

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
  
  uint64_t max_bytes = maxlat * maxlat * maxlat * Ls * sizeof(HalfSpinColourVectorD);

  // locates xbuf with maximum bytes upfront
  std::vector<HalfSpinColourVectorD *> xbuf(8);
  std::vector<HalfSpinColourVectorD *> rbuf(8);
  for (int d = 0; d < 8; d++)
  {
    xbuf[d] = (HalfSpinColourVectorD *)acceleratorAllocDevice(max_bytes);
    rbuf[d] = (HalfSpinColourVectorD *)acceleratorAllocDevice(max_bytes);
  }

  for (int lat = 16; lat <= maxlat; lat += 8)
  {

    Coordinate latt_size({lat * mpi_layout[0], lat * mpi_layout[1], lat * mpi_layout[2],
                          lat * mpi_layout[3]});

    GridCartesian Grid(latt_size, simd_layout, mpi_layout);
    RealD Nrank = Grid._Nprocessors;
    RealD Nnode = Grid.NodeCount();
    RealD ppn = Nrank / Nnode;

    uint64_t bytes = lat * lat * lat * Ls * sizeof(HalfSpinColourVectorD);
    for (int d = 0; d < 8; d++)
    {
      acceleratorMemSet(xbuf[d], 0, bytes);
      acceleratorMemSet(rbuf[d], 0, bytes);
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
  }
  for (int d = 0; d < 8; d++)
  {
    acceleratorFreeDevice(xbuf[d]);
    acceleratorFreeDevice(rbuf[d]);
  }
  return;
}
