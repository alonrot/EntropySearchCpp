// Copyright 2020 Max Planck Society. All rights reserved.
// 
// Author: Alonso Marco Valle (amarcovalle/alonrot) amarco(at)tuebingen.mpg.de
// Affiliation: Max Planck Institute for Intelligent Systems, Autonomous Motion
// Department / Intelligent Control Systems
// 
// This file is part of EntropySearchCpp.
// 
// EntropySearchCpp is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
// 
// EntropySearchCpp is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.
// 
// You should have received a copy of the GNU General Public License along with
// EntropySearchCpp.  If not, see <http://www.gnu.org/licenses/>.
//
//
// Imported manually from thirparties/lwafomps/ (math tools library)

#include <cmath>
#include "erfinv.h"

#if defined(MAIN)
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#endif

using namespace std;
namespace  {
#if !defined(ERFINV2) && !defined(ERFINV3)
    /*
      constants from Information Theory and Signal Processing Library (libit):
      http://libit.sourceforge.net/
      erfinv in math.c
     */
    const double aa0 = 0.886226899;
    const double aa1 = -1.645349621;
    const double aa2 = 0.914624893;
    const double aa3 = -0.140543331;
    const double bb0 = 1.0;
    const double bb1 = -2.118377725;
    const double bb2 = 1.442710462;
    const double bb3 = -0.329097515;
    const double bb4 = 0.012229801;
    const double cc0 = -1.970840454;
    const double cc1 = -1.62490649;
    const double cc2 = 3.429567803;
    const double cc3 = 1.641345311;
    const double dd0 = 1.0;
    const double dd1 = 3.543889200;
    const double dd2 = 1.637067800;
    const double sqrtPI = sqrt(M_PI);
#endif
}

namespace ErrorFunction {
#if !defined(ERFINV2) && !defined(ERFINV3)
    /*
      erfinv() is a modification of the function erfinv() in
      Information Theory and Signal Processing Library (libit):
      http://libit.sourceforge.net/
      erfinv() in math.c
      LICENSE: GNU LIBRARY GENERAL PUBLIC LICENSE Version 2, June 1991
      see LICENSE.

      modified by Mutsuo Saito, 2015.
    */
    double erfinv(double x)
    {
        double  r, y;
        int  sign_x;

        if (x < -1 || x > 1) {
            return NAN;
        }
        if (x == 0.0) {
            return 0.0;
        }
        if (x > 0) {
            sign_x = 1;
        } else {
            sign_x = -1;
            x = -x;
        }
        if (x <= 0.7)  {
            double x2 = x * x;
            r = x * (((aa3 * x2 + aa2) * x2 + aa1) * x2 + aa0);
            r /= (((bb4 * x2 + bb3) * x2 + bb2) * x2 + bb1) * x2 + bb0;
        } else {
            y = sqrt (-log ((1.0 - x) / 2.0));
            r = (((cc3 * y + cc2) * y + cc1) * y + cc0);
            r /= ((dd2 * y + dd1) * y + dd0);
        }
        r = r * sign_x;
        x = x * sign_x;
        r -= (erf (r) - x) / (2 / sqrtPI * exp (-r * r));
        r -= (erf (r) - x) / (2 / sqrtPI * exp (-r * r));
        return r;
    }
#elif defined(ERFINV2)
    /**
     * copied from erfinv_SP_1.cu by Prof. Mike Giles.
     * https://people.maths.ox.ac.uk/gilesm/
     * https://people.maths.ox.ac.uk/gilesm/codes/erfinv/
     *
     * original code is written for CUDA and for single precision.
     * modified by Mutsuo Saito in 2016.
     */
    double erfinv(double x)
    {
        double w, p;
        double sign;
        if (x >= 0) {
            sign = 1.0;
        } else {
            sign = -1.0;
            x = abs(x);
        }
        w = - log((1.0 - x) * (1.0 + x));
        if (w < 5.0) {
            w = w - 2.5;
            p = 2.81022636e-08;
            p = 3.43273939e-07 + p*w;
            p = -3.5233877e-06 + p*w;
            p = -4.39150654e-06 + p*w;
            p = 0.00021858087 + p*w;
            p = -0.00125372503 + p*w;
            p = -0.00417768164 + p*w;
            p = 0.246640727 + p*w;
            p = 1.50140941 + p*w;
        } else {
            w = sqrt(w) - 3.000000;
            p = -0.000200214257;
            p = 0.000100950558 + p*w;
            p = 0.00134934322 + p*w;
            p = -0.00367342844 + p*w;
            p = 0.00573950773 + p*w;
            p = -0.0076224613 + p*w;
            p = 0.00943887047 + p*w;
            p = 1.00167406 + p*w;
            p = 2.83297682 + p*w;
        }
        return sign * p * x;
    }
#else // ERFINV3
    /**
     * copied from erfinv_DP_1.cu by Prof. Mike Giles.
     * https://people.maths.ox.ac.uk/gilesm/
     * https://people.maths.ox.ac.uk/gilesm/codes/erfinv/
     *
     * Original code is written for CUDA.
     * Mutsuo Saito modified original code for C++.
     */
    double erfinv(double x)
    {
        double w, p;
        double sign;
        if (x > 0) {
            sign = 1.0;
        } else {
            sign = -1.0;
            x = abs(x);
        }
        w = - log((1.0-x)*(1.0+x));

        if ( w < 6.250000 ) {
            w = w - 3.125000;
            p =  -3.6444120640178196996e-21;
            p =   -1.685059138182016589e-19 + p*w;
            p =   1.2858480715256400167e-18 + p*w;
            p =    1.115787767802518096e-17 + p*w;
            p =   -1.333171662854620906e-16 + p*w;
            p =   2.0972767875968561637e-17 + p*w;
            p =   6.6376381343583238325e-15 + p*w;
            p =  -4.0545662729752068639e-14 + p*w;
            p =  -8.1519341976054721522e-14 + p*w;
            p =   2.6335093153082322977e-12 + p*w;
            p =  -1.2975133253453532498e-11 + p*w;
            p =  -5.4154120542946279317e-11 + p*w;
            p =    1.051212273321532285e-09 + p*w;
            p =  -4.1126339803469836976e-09 + p*w;
            p =  -2.9070369957882005086e-08 + p*w;
            p =   4.2347877827932403518e-07 + p*w;
            p =  -1.3654692000834678645e-06 + p*w;
            p =  -1.3882523362786468719e-05 + p*w;
            p =    0.0001867342080340571352 + p*w;
            p =  -0.00074070253416626697512 + p*w;
            p =   -0.0060336708714301490533 + p*w;
            p =      0.24015818242558961693 + p*w;
            p =       1.6536545626831027356 + p*w;
        }
        else if ( w < 16.000000 ) {
            w = sqrt(w) - 3.250000;
            p =   2.2137376921775787049e-09;
            p =   9.0756561938885390979e-08 + p*w;
            p =  -2.7517406297064545428e-07 + p*w;
            p =   1.8239629214389227755e-08 + p*w;
            p =   1.5027403968909827627e-06 + p*w;
            p =   -4.013867526981545969e-06 + p*w;
            p =   2.9234449089955446044e-06 + p*w;
            p =   1.2475304481671778723e-05 + p*w;
            p =  -4.7318229009055733981e-05 + p*w;
            p =   6.8284851459573175448e-05 + p*w;
            p =   2.4031110387097893999e-05 + p*w;
            p =   -0.0003550375203628474796 + p*w;
            p =   0.00095328937973738049703 + p*w;
            p =   -0.0016882755560235047313 + p*w;
            p =    0.0024914420961078508066 + p*w;
            p =   -0.0037512085075692412107 + p*w;
            p =     0.005370914553590063617 + p*w;
            p =       1.0052589676941592334 + p*w;
            p =       3.0838856104922207635 + p*w;
        }
        else {
            w = sqrt(w) - 5.000000;
            p =  -2.7109920616438573243e-11;
            p =  -2.5556418169965252055e-10 + p*w;
            p =   1.5076572693500548083e-09 + p*w;
            p =  -3.7894654401267369937e-09 + p*w;
            p =   7.6157012080783393804e-09 + p*w;
            p =  -1.4960026627149240478e-08 + p*w;
            p =   2.9147953450901080826e-08 + p*w;
            p =  -6.7711997758452339498e-08 + p*w;
            p =   2.2900482228026654717e-07 + p*w;
            p =  -9.9298272942317002539e-07 + p*w;
            p =   4.5260625972231537039e-06 + p*w;
            p =  -1.9681778105531670567e-05 + p*w;
            p =   7.5995277030017761139e-05 + p*w;
            p =  -0.00021503011930044477347 + p*w;
            p =  -0.00013871931833623122026 + p*w;
            p =       1.0103004648645343977 + p*w;
            p =       4.8499064014085844221 + p*w;
        }
        return sign * p * x;
    }
#endif
}
