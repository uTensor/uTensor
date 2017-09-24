##Finding your target name

`mbed detect` to see which target is connect to the board

`mbedls -l` to list all supported targets

##Build Steps

`mbed compile -m K64F --profile=./build_profile/release.json` to build for K64F

`mbed compile -m K64F --profile=./build_profile/release.json -f` to compile and flash

##Configure

See mbed_app.json

##Others

You will need a SD Card (and a reader if not ready built into the board)
