#!/bin/bash

# Train

chmod 777 track1_run.sh

bash track1_run.sh

mv student.pth track_1_pth_1.pth

bash track1_run.sh

mv student.pth track_1_pth_2.pth

bash track1_run.sh

mv student.pth track_1_pth_3.pth


