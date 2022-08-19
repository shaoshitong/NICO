#!/bin/bash

# Train

chmod 777 track2_run.sh

bash track2_run.sh

mv student.pth track_2_pth_1.pth

bash track2_run.sh

mv student.pth track_2_pth_2.pth

bash track2_run.sh

mv student.pth track_2_pth_3.pth
