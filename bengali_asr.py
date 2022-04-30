import os
from weakref import KeyedRef
import wget
import zipfile
import argparse
import subprocess

from deepspeech_pytorch.data.data_opts import add_data_opts
from tqdm import tqdm
import shutil

from deepspeech_pytorch.data.utils import create_manifest

parser = argparse.ArgumentParser(description='Processes and downloads LibriSpeech dataset.')
parser = add_data_opts(parser)
parser.add_argument("--target-dir", default='Bengali_ASR_dataset/', type=str, help="Directory to store the dataset.")
parser.add_argument('--files-to-use', default="asr_bengali_0.zip,"
                                              "asr_bengali_1.zip,"
                                              "asr_bengali_2.zip,"
                                              "asr_bengali_3.zip,"
                                              "asr_bengali_4.zip,"
                                              "asr_bengali_5.zip,"
                                              "asr_bengali_6.zip,"
                                              "asr_bengali_7.zip,"
                                              "asr_bengali_8.zip,"
                                              "asr_bengali_9.zip,"
                                              "asr_bengali_a.zip,"
                                              "asr_bengali_b.zip,"
                                              "asr_bengali_c.zip,"
                                              "asr_bengali_d.zip,"
                                              "asr_bengali_e.zip,"
                                              "asr_bengali_f.zip", type=str,
                    help='list of file names to download')
args = parser.parse_args()

BANGLA_ASR_URLS = {
    "train": ["https://www.openslr.org/resources/53/asr_bengali_0.zip",
              "https://www.openslr.org/resources/53/asr_bengali_1.zip",
              "https://www.openslr.org/resources/53/asr_bengali_2.zip",
              "https://www.openslr.org/resources/53/asr_bengali_3.zip",
              "https://www.openslr.org/resources/53/asr_bengali_4.zip",
              "https://www.openslr.org/resources/53/asr_bengali_5.zip",
              "https://www.openslr.org/resources/53/asr_bengali_6.zip",
              "https://www.openslr.org/resources/53/asr_bengali_7.zip",
              "https://www.openslr.org/resources/53/asr_bengali_8.zip",
              "https://www.openslr.org/resources/53/asr_bengali_9.zip",
              "https://www.openslr.org/resources/53/asr_bengali_a.zip",
              "https://www.openslr.org/resources/53/asr_bengali_b.zip",
              "https://www.openslr.org/resources/53/asr_bengali_c.zip"],

    "val": ["https://www.openslr.org/resources/53/asr_bengali_e.zip",
            "https://www.openslr.org/resources/53/asr_bengali_d.zip"],

    "test_clean": ["https://www.openslr.org/resources/53/asr_bengali_f.zip"],
}


def _preprocess_transcript(phrase):
    return phrase.strip()


def _process_file(wav_dir, txt_dir, base_filename, root_dir, transcriptions):
    full_recording_path = os.path.join(root_dir, base_filename)
    assert os.path.exists(full_recording_path) and os.path.exists(root_dir)
    wav_recording_path = os.path.join(wav_dir, base_filename.replace(".flac", ".wav"))

    # process transcript
    txt_transcript_path = os.path.join(txt_dir, base_filename.replace(".flac", ".txt"))

    with open(txt_transcript_path, "w", encoding='utf8') as f:
        key = base_filename.replace(".flac", "")
        if key in transcriptions:
          f.write(_preprocess_transcript(transcriptions[key]))
          subprocess.call(["sox {}  -r {} -b 16 -c 1 {}".format(full_recording_path, str(args.sample_rate),
                                                          wav_recording_path)], shell=True)
        f.flush()
        


def main():
    target_dl_dir = args.target_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)
    files_to_dl = args.files_to_use.strip().split(',')

    files_to_dl = args.files_to_use.strip().split(',')
    for split_type, lst_bangla_asr_urls in BANGLA_ASR_URLS.items():
        split_dir = os.path.join(target_dl_dir, split_type)
        
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        split_wav_dir = os.path.join(split_dir, "wav")
        if not os.path.exists(split_wav_dir):
            os.makedirs(split_wav_dir)
        split_txt_dir = os.path.join(split_dir, "txt")
        if not os.path.exists(split_txt_dir):
            os.makedirs(split_txt_dir)
        extracted_dir = os.path.join(split_dir, "asr_bengali")
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)
        
        transcript_file = os.path.join('/content/drive/MyDrive/NSU/CSE499A/SeanNaren/Final_DeepSpeech2/deepspeech.pytorch/data/Bengali_ASR_dataset', 'utt_spk_text.tsv')
        assert os.path.exists(transcript_file), "Transcript file {} does not exist.".format(transcript_file)
        transcriptions = open(transcript_file).read().strip().split("\n")
        transcriptions = {t.split()[0].split("\t")[-1]: " ".join(t.split()[2:]) for t in transcriptions}

        for url in lst_bangla_asr_urls:
            # check if we want to dl this file
            dl_flag = False
            for f in files_to_dl:
                if url.find(f) != -1:
                    dl_flag = True
            if not dl_flag:
                print("Skipping url: {}".format(url))
                continue
            filename = url.split("/")[-1]
            target_filename = os.path.join(split_dir, filename)
            if not os.path.exists(target_filename):
                wget.download(url, split_dir)
            
            print("Unpacking {}...".format(filename))          
            with zipfile.ZipFile(target_filename, 'r') as zip_ref:
              zip_ref.extractall(split_dir)
            os.remove(target_filename)
            
            print("Converting flac files to wav and extracting transcripts...")
            assert os.path.exists(extracted_dir), "Archive {} was not properly uncompressed.".format(filename)
            
            for root, subdirs, files in tqdm(os.walk(extracted_dir)):
                for f in files:
                    if f.find(".flac") != -1:
                        _process_file(wav_dir=split_wav_dir, txt_dir=split_txt_dir,
                                      base_filename=f, root_dir=root,
                                      transcriptions=transcriptions)

            print("Finished {}".format(url))
            shutil.rmtree(extracted_dir)
        
        if split_type == 'train':  # Prune to min/max duration
            create_manifest(
                data_path=split_dir,
                output_name='bengali_asr_' + split_type + '_manifest.json',
                manifest_path=args.manifest_dir,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                num_workers=args.num_workers
            )
        else:
            create_manifest(
                data_path=split_dir,
                output_name='bengali_asr_' + split_type + '_manifest.json',
                manifest_path=args.manifest_dir,
                num_workers=args.num_workers
            )
            



if __name__ == "__main__":
    main()
