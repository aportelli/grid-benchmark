/*
Copyright © 2022 Antonin Portelli <antonin.portelli@me.com>

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

#ifndef Grid_Benchmarks_Common_hpp_
#define Grid_Benchmarks_Common_hpp_

#ifndef GRID_MSG
#define GRID_MSG std::cout << GridLogMessage
#endif

#ifndef GRID_MSG_MAXSIZE
#define GRID_MSG_MAXSIZE 1024
#endif

#define GRID_BIG_SEP                                                                     \
  "==============================================================================="
#define GRID_SMALL_SEP "------------------------------------------"

#define grid_big_sep()                                                                   \
  {                                                                                      \
    GRID_MSG << GRID_BIG_SEP << std::endl;                                               \
  }

#define grid_small_sep()                                                                 \
  {                                                                                      \
    GRID_MSG << GRID_SMALL_SEP << std::endl;                                             \
  }

#define grid_printf(...)                                                                 \
  {                                                                                      \
    char _buf[GRID_MSG_MAXSIZE];                                                         \
    snprintf(_buf, GRID_MSG_MAXSIZE, __VA_ARGS__);                                       \
    GRID_MSG << _buf;                                                                    \
  }

#endif // Grid_Benchmarks_Common_hpp_
