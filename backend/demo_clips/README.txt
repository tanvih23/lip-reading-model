Demo clips for Member D — upload / predict testing
================================================

12 short videos, one per word (from GRID corpus).
Each clip shows only that word being spoken (+ 5 frames context).

Files:
  demo_bin.mp4, demo_blue.mp4, demo_red.mp4, demo_white.mp4
  demo_lay.mp4, demo_place.mp4, demo_set.mp4
  demo_again.mp4, demo_now.mp4, demo_please.mp4, demo_soon.mp4
  demo_green.mp4

Test preprocessing:
  python preprocess_upload.py demo_clips\demo_bin.mp4

Expected: shape (20, 96, 96)

Use these in your /predict endpoint upload tests before the live demo.
