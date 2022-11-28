echo "This is shell script to run demucs with given arguments"

$path  echo "test/path"
echo $path

python.exe -m demucs --wav E:\Libri2Mix\wav8k\max --musdb E:\Libri2Mix\wav8k\max --samplerate 8000 --audio_channels 1 --no_augment --workers 4 --repitch 0
