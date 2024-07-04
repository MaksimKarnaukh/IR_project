# PyLucene Installation Guide

This is a quick guide on how to get pyLucene up and running on ubuntu.

### Dependencies

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install 
sudo apt-get install openjdk-11-jdk ant build-essential python3-dev python3-distutils python3-pip
```

### JCC

Download pylucene 9.10.0 from [here](https://dlcdn.apache.org/lucene/pylucene/?C=M;O=A) and place it somewhere, e.g. the libs folder.
Ensure there is a Makefile in the `pylucene-9.10.0/` folder.

```bash
cd pylucene-9.10.0
cd jcc
```

Edit the `setup.py` file (e.g. `sudo nano setup.py`), change the JDK 'linux' value to the path of your JDK installation:

```python
JDK = {
    ...,
    'linux': '/usr/lib/jvm/java-11-openjdk-amd64',
    ...
}
```

Create two symlinks (for the `libjava.so` and `libjvm.so` files) needed for the setup.py script:

```bash
sudo ln -s /usr/lib/jvm/java-11-openjdk-amd64/lib/libjava.so /usr/lib/libjava.so
sudo ln -s /usr/lib/jvm/java-11-openjdk-amd64/lib/server/libjvm.so /usr/lib/libjvm.so
```

After that, run the following commands:

```bash
sudo python3 setup.py build
sudo python3 setup.py install
```

### PyLucene

Navigate to the `pylucene-9.10.0/` folder:

```bash
cd ..
```

We need to (if needed) edit the `Makefile` file (e.g. `sudo nano Makefile`) for our configuration, change the variables (somewhere around line 85 where there are four commented lines each time with a description comment above):

```makefile
PREFIX_PYTHON=/usr
ANT=ant
PYTHON=$(PREFIX_PYTHON)/bin/python3
JCC=$(PYTHON) -m jcc --shared
NUM_FILES=16
```

Now, run the following commands:

```bash
make
make test
sudo make install
```

### Testing

To test if everything is working correctly, run the following command:

```bash
python3
>>import lucene
>>lucene.initVM()
  <jcc.JCCEnv object at 0x7fed9a8acf30>
>>print(lucene.VERSION)
  9.10.0
```
