
# 4371 tracks
# walltime is 48h tops
# tracks take 3-10m to process, say 8min on average
# that means 12.1 workers minimum in expectation; we do 16 to be safe
# 4371/20 = 218.55 so we do 220 chunks

qsub -v start='0',end='220' process_all_tracks.sh
qsub -v start='220',end='440' process_all_tracks.sh
qsub -v start='440',end='660' process_all_tracks.sh
qsub -v start='660',end='880' process_all_tracks.sh
qsub -v start='880',end='1100' process_all_tracks.sh
qsub -v start='1100',end='1320' process_all_tracks.sh
qsub -v start='1320',end='1540' process_all_tracks.sh
qsub -v start='1540',end='1760' process_all_tracks.sh
qsub -v start='1760',end='1980' process_all_tracks.sh
qsub -v start='1980',end='2200' process_all_tracks.sh
qsub -v start='2200',end='2420' process_all_tracks.sh
qsub -v start='2420',end='2640' process_all_tracks.sh
qsub -v start='2640',end='2860' process_all_tracks.sh
qsub -v start='2860',end='3080' process_all_tracks.sh
qsub -v start='3080',end='3300' process_all_tracks.sh
qsub -v start='3300',end='3520' process_all_tracks.sh
qsub -v start='3520',end='3740' process_all_tracks.sh
qsub -v start='3740',end='3960' process_all_tracks.sh
qsub -v start='3960',end='4180' process_all_tracks.sh
qsub -v start='4180',end='-1' process_all_tracks.sh
