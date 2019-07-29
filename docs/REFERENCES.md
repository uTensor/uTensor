## The Basic

- [How to Quantize Neural Network with Tensorflow](https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/)
  - [Why Are 8 Bits Enough for Deep Neural Network](https://petewarden.com/2015/05/23/why-are-eight-bits-enough-for-deep-neural-networks/)
- [tf source code](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/kernels)
- [mbed openocd](https://docs.mbed.com/docs/mbed-os-handbook/en/5.4/debugging/toolchain/)

## How to Run `gdb` with OpenOCD

1. Start server
  - connect your board
  - run `openocd` to start the server
  - [doc](https://docs.mbed.com/docs/mbed-os-handbook/en/latest/debugging/toolchain/)
2. run `arm-none-eabi-gdb` with `uTensor.elf`.
  - `arm-none-eabi-gdb PATH/TO/ELF/uTensor.elf`
  - in the `gdb` console, type `target remote localhost:3333`
3. setup break point, next line, ...etc
  - happy `gdb`ing! 

## Telnet with OpenOCD

- telnet localhost 4444
- halt
- flash probe 0
- flash write_image erase uTensor.bin 0x08000000
  - You have to run openocd server in the directory of uTensor.bin
- reset
- exit

## Misc

- [idx format detail](http://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html)
