from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Microsoft XP* fields (UTF-16LE encoded)
XP_TAGS = {
    0x9C9B: "XPTitle",
    0x9C9C: "XPComment",
    0x9C9D: "XPAuthor",
    0x9C9E: "XPKeywords",
    0x9C9F: "XPSubject",
}

# Tags that point to SubIFDs
IFD_POINTERS = {
    0x8769: "Exif IFD",
    0x8825: "GPS IFD",
    0xA005: "Interop IFD",
    0x014A: "SubIFDs",  # can contain multiple
}

def decode_value(tag_id, value):
    """Decode XP* strings, leave others as-is."""
    if tag_id in XP_TAGS and isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-16-le").rstrip("\x00")
        except Exception:
            return value
    return value

def parse_ifd(exif_data, ifd_name="IFD0"):
    result = {}

    for tag_id, raw_value in exif_data.items():
        # Choose name mapping depending on IFD type
        if ifd_name.startswith("GPS"):
            tag_name = GPSTAGS.get(tag_id, f"Unknown-0x{tag_id:04X}")
        else:
            tag_name = TAGS.get(tag_id, XP_TAGS.get(tag_id, f"Unknown-0x{tag_id:04X}"))

        value = decode_value(tag_id, raw_value)
        result[tag_name] = value

        # Follow SubIFDs
        if tag_id in IFD_POINTERS and isinstance(raw_value, int):
            try:
                subifd = exif_data.get_ifd(tag_id)
                result[IFD_POINTERS[tag_id]] = parse_ifd(subifd, IFD_POINTERS[tag_id])
            except Exception:
                pass

    return result

def get_exif_dict(image_path):
    """Return all EXIF + SubIFDs as a nested dictionary."""
    with Image.open(image_path) as im:
        exif = im.getexif()
        if not exif:
            return {}
        return parse_ifd(exif, "IFD0")

# Example usage
# meta = get_exif_dict(r"C:\Users\PRYth\Downloads\Prime_Minister_of_Singapore_Lawrence_Wong_250530-D-PM193-4275_(2025).jpg")
# print(meta)
# if "photoshop" in meta.keys():
#     print("photoshopped")
# else:
#     print("nein photoshopped")
