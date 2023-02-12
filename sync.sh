#!/bin/bash
rsync -av /home/drudis/GH/Julien/scripts_lap/  ubuntu@9.4.130.227:/home/ubuntu/scripts_laptop/
rsync -av ubuntu@9.4.130.227:/home/ubuntu/scripts_laptop/ /home/drudis/GH/Julien/scripts_lap/