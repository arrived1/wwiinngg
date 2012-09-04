#include <iostream>


int main()
{
    //mass, x, y, z, v.x, v.y, v.z, 

    int const particles = 8192;

    for(int i = -50; i < -30; ++i)
        for(int j = -10; j < 10; ++j)
            for(int k = -10; k < 10; ++k)
                std::cout << 4.98914e-05 << "\t" 
                          << i << "\t" << j << "\t" << k 
                          << "\t" << 0.5 << "\t" << 0.0 << "\t" << 0.0 
                          << std::endl;

    //for(int i = 0; i < 192; ++i)
        //std::cout << 4.98914e-05 << "\t" 
                  //<< i + 100<< "\t" << i + 110<< "\t" << i + 101
                  //<< "\t" << 0.0 << "\t" << 0.1 << "\t" << 0.0 
                  //<< std::endl;
    
    //for(int i = 0; i < particles; ++i)
        //std::cout << 4.98914e-05 << "\t" 
                  //<< i << "\t" << 0.0 << "\t" << 0.0
                  //<< "\t" << 0.0 << "\t" << 0.1 << "\t" << 0.0 
                  //<< std::endl;
}
