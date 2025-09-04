import ffmpeg
import json # For pretty printing the output

def get_video_metadata(file_path):
    """
    Extracts and returns metadata from a video file.

    Args:
        file_path (str): The path to the video file.

    Returns:
        dict: A dictionary containing the video's metadata,
                or None if an error occurs.
    """
    try:
        probe = ffmpeg.probe(file_path)
        return probe
    except ffmpeg.Error as e:
        print(f"Error extracting metadata: {e.stderr.decode()}")
        return None

if __name__ == "__main__":
    video_file = r"C:\Users\PRYth\OneDrive\Desktop\Agentic\SimplifyAgentic\test_input\Singlish. u wan cock or flu joos _.mp3"  # Replace with the path to your video file
    metadata = get_video_metadata(video_file)

    if metadata:
        print(json.dumps(metadata, indent=4))
        # Access specific metadata, for example:
        # print(f"Format Duration: {metadata['format']['duration']} seconds")
        # print(f"Video Stream Codec: {metadata['streams'][0]['codec_name']}")
    else:
        print(f"Failed to extract metadata for {video_file}")