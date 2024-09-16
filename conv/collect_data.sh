#!/bin/bash

bash ./compile.sh

cd build/

make

./main -i ../samples/3,5.pgm      -o "../outputs/s"
./main -i ../samples/JENNYROM.pgm -o "../outputs/m"
./main -i ../samples/LOG.pgm      -o "../outputs/l"
./main -i ../samples/longcol.pgm  -o "../outputs/c"
./main -i ../samples/longrow.pgm  -o "../outputs/r"
./pgm_creator 1 1       "../samples/pixel.pgm"
./pgm_creator 2048 2048 "../samples/square.pgm"
./pgm_creator 1048576 4 "../samples/wide.pgm"
./pgm_creator 4 1048576 "../samples/tall.pgm"

rm ../outputs/*

for i in {1..100}
do
	echo "S"; ./main -i ../samples/3,5.pgm      -o ../outputs/s >> "../outputs/s.txt"
	echo "M"; ./main -i ../samples/JENNYROM.pgm -o ../outputs/m >> "../outputs/m.txt"
	echo "L"; ./main -i ../samples/LOG.pgm      -o ../outputs/l >> "../outputs/l.txt"
	echo "C"; ./main -i ../samples/longcol.pgm  -o ../outputs/c >> "../outputs/c.txt"
	echo "R"; ./main -i ../samples/longrow.pgm  -o ../outputs/r >> "../outputs/r.txt"
	echo "P"; ./main -i ../samples/pixel.pgm    -o ../outputs/pixel   >> "../outputs/pixel.txt"
	echo "GS"; ./main -i ../samples/square.pgm   -o ../outputs/square >> "../outputs/square.txt"
	echo "GW"; ./main -i ../samples/wide.pgm   -o ../outputs/wide >> "../outputs/wide.txt"
	echo "GT"; ./main -i ../samples/tall.pgm   -o ../outputs/tall >> "../outputs/tall.txt"
done

cd ..
python3 generate_graphs.py --filename "s";
python3 generate_graphs.py --filename "m";
python3 generate_graphs.py --filename "l";
python3 generate_graphs.py --filename "c";
python3 generate_graphs.py --filename "r";
python3 generate_graphs.py --filename "pixel";
python3 generate_graphs.py --filename "square";
python3 generate_graphs.py --filename "wide";
python3 generate_graphs.py --filename "tall";
#rm -r build/
