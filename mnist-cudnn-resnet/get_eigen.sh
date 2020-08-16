#!/bin/bash

echo "Delete the eigen dir ..."
rm -rf eigen

echo "Download and extract the eigen ..."
if [ ! -f "49177915a14a.tar.gz" ]; then
  wget http://10.201.40.22/files/eigen/49177915a14a.tar.gz
fi
tar -xvf 49177915a14a.tar.gz > /dev/null
mv eigen-eigen-49177915a14a eigen

echo "Apply the eigen patch ..."
cd eigen
patch -p1 < ../eigen.patch

if [ $? == 0 ]; then
  echo "Done"
else
  echo "Error occured"
fi
