#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


int main()
{
    srand(time(NULL));

    int bodies = 0;
    float step = 0.2;
    float mass = 1;

    for(float i = -30.0; i < -26.0; i += step)
        for(float j = -1.0; j < 1.0; j += step)
            for(float k = -3.0; k < 3.0; k += step)
            {
                bodies++;
                //std::cout << "[" << i << ", " << j << ", " << k << "]" << std::endl;
                std::cout << mass << "\t"
                          << i << "\t" << j << "\t" << k << "\t"
                          << fabs(float((rand() % 30) - 10.f) / 10.f) << "\t" << 0.0 << "\t" << 0.0
                          << std::endl;
            }
}












// #include <iostream>
// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>

// int main()
// {
//   srand(time(NULL));
// //collision test
//      std::cout << 4.98914e-05 << "\t" 
//                   << 10 << "\t" << 30 << "\t" << 0
//                   << "\t" << 0.5 << "\t" << 0.0 << "\t" << 0.0 
//                   << std::endl;

//       std::cout << 4.98914e-05 << "\t" 
//               << 20 << "\t" << 30 << "\t" << 0
//               << "\t" << -0.5 << "\t" << 0.0 << "\t" << 0.0 
//               << std::endl;

// ///normal data initialization
//     int const particles = 8192;

//     for(int i = -30; i < -11; ++i)
//         for(int j = -10; j < 10; ++j)
//             for(int k = -10; k < 10; ++k)
//                 std::cout << 4.98914e-05 << "\t" 
//                           << i << "\t" << j << "\t" << k 
//                           << "\t" << float((rand()%30) - 10.f) / 10.f << "\t" << 0.0 << "\t" << 0.0 
//                           << std::endl;

//     //random
//     //float x = float((rand()%50) - 0.f)*10.0f;

//     //for(int i = 0; i < 192; ++i)
//         //std::cout << 4.98914e-05 << "\t" 
//                   //<< i + 100<< "\t" << i + 110<< "\t" << i + 101
//                   //<< "\t" << 0.0 << "\t" << 0.1 << "\t" << 0.0 
//                   //<< std::endl;

//     //for(int i = 0; i < particles; ++i)
//         //std::cout << 4.98914e-05 << "\t" 
//                   //<< i << "\t" << 0.0 << "\t" << 0.0
//                   //<< "\t" << 0.0 << "\t" << 0.1 << "\t" << 0.0 
//                   //<< std::endl;
//     }
