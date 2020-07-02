from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def generate_segmented_video(vid_info, segment_data, video_root, save_root=None, verbose=0):
    video_name = vid_info['vid_name'][0]
    video_path = video_root + '/' + video_name
    video_name = video_name.replace(".mp4", "")

    if save_root is None:
        save_root = video_root + "/" + "segment/"

    video_sync = segment_data['video_sync_time']

    if verbose > 0:
        print("Video sync time :: ", video_sync, "s", end="")

    for i in range(segment_data['segment_x'].shape[0]):
        video_x = segment_data['segment_x'][i]
        segment_name = segment_data['segment_name'][i]
        segment_label = segment_data['segment_label'][i]

        start_t = video_x[0] + video_sync
        end_t = video_x[video_x.shape[0] - 1] + video_sync

        file_name = video_name + "_" + str(i) + "_" + str(segment_label) + "_" + segment_name + ".mp4"

        ffmpeg_extract_subclip(video_path, start_t, end_t, save_root + file_name)

        if verbose > 0:
            print(file_name, "::", start_t, "~", end_t)
