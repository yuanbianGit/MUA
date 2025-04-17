source /data1/by/miniconda3/etc/profile.d/conda.sh
conda activate by

cd /data4/by/reid/github/TOP-ReID-master
python att_mma.py --config_file ./configs/RGBNT201/ATT_TOP.yml MODEL.DEVICE_ID "('1')" SOLVER.BASE_LR 0.0006 OUTPUT_DIR ./ATT_LOG/base_0.0006
python att_mma.py --config_file ./configs/RGBNT201/ATT_TOP.yml MODEL.DEVICE_ID "('1')" SOLVER.BASE_LR 0.0008 OUTPUT_DIR ./ATT_LOG//base_0.0008
python att_mma.py --config_file ./configs/RGBNT201/ATT_TOP.yml MODEL.DEVICE_ID "('1')" SOLVER.BASE_LR 0.0010 OUTPUT_DIR ./ATT_LOG//base_0.0010