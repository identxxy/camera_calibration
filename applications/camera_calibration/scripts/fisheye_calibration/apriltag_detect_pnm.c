// Lightweight AprilTag detector wrapper used by prepare_fisheye_intrinsics_from_mcap.py.

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "apriltag.h"
#include "common/pnm.h"
#include "common/zarray.h"
#include "tag36h11.h"


static uint8_t scaled_sample(const uint8_t* data, int index, int max_value) {
  if (max_value == 255) {
    return data[index];
  }
  return (uint8_t) (((int) data[index] * 255) / max_value);
}


static uint8_t* make_gray_image(const pnm_t* pnm) {
  uint8_t* gray = (uint8_t*) malloc((size_t) pnm->width * (size_t) pnm->height);
  if (!gray) {
    return NULL;
  }

  if (pnm->format == PNM_FORMAT_GRAY) {
    for (int i = 0; i < pnm->width * pnm->height; ++i) {
      gray[i] = scaled_sample(pnm->buf, i, pnm->max);
    }
  } else if (pnm->format == PNM_FORMAT_RGB) {
    for (int i = 0; i < pnm->width * pnm->height; ++i) {
      int base = 3 * i;
      int r = scaled_sample(pnm->buf, base + 0, pnm->max);
      int g = scaled_sample(pnm->buf, base + 1, pnm->max);
      int b = scaled_sample(pnm->buf, base + 2, pnm->max);
      gray[i] = (uint8_t) ((77 * r + 150 * g + 29 * b) >> 8);
    }
  } else {
    free(gray);
    return NULL;
  }

  return gray;
}


int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s image.pgm [quad_decimate] [bits_corrected]\\n", argv[0]);
    return 2;
  }

  float quad_decimate = 1.0f;
  if (argc >= 3) {
    quad_decimate = (float) atof(argv[2]);
  }
  int bits_corrected = 2;
  if (argc >= 4) {
    bits_corrected = atoi(argv[3]);
  }

  pnm_t* pnm = pnm_create_from_file(argv[1]);
  if (!pnm) {
    fprintf(stderr, "Cannot read PNM image: %s\\n", argv[1]);
    return 3;
  }

  uint8_t* gray = make_gray_image(pnm);
  if (!gray) {
    fprintf(stderr, "Unsupported PNM image format: %s\\n", argv[1]);
    pnm_destroy(pnm);
    return 4;
  }

  image_u8_t image = {
    .width = pnm->width,
    .height = pnm->height,
    .stride = pnm->width,
    .buf = gray,
  };

  apriltag_family_t* family = tag36h11_create();
  apriltag_detector_t* detector = apriltag_detector_create();
  detector->nthreads = 1;
  detector->quad_decimate = quad_decimate;
  detector->quad_sigma = 0.0f;
  detector->refine_edges = 1;
  apriltag_detector_add_family_bits(detector, family, bits_corrected);

  zarray_t* detections = apriltag_detector_detect(detector, &image);
  int tag_count = zarray_size(detections);

  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_margin = 0.0;
  double min_x = DBL_MAX;
  double min_y = DBL_MAX;
  double max_x = -DBL_MAX;
  double max_y = -DBL_MAX;

  for (int i = 0; i < tag_count; ++i) {
    apriltag_detection_t* detection = NULL;
    zarray_get(detections, i, &detection);
    sum_x += detection->c[0];
    sum_y += detection->c[1];
    sum_margin += detection->decision_margin;
    for (int j = 0; j < 4; ++j) {
      if (detection->p[j][0] < min_x) min_x = detection->p[j][0];
      if (detection->p[j][1] < min_y) min_y = detection->p[j][1];
      if (detection->p[j][0] > max_x) max_x = detection->p[j][0];
      if (detection->p[j][1] > max_y) max_y = detection->p[j][1];
    }
  }

  if (tag_count == 0) {
    printf("{\"tag_count\":0,\"centroid_x\":null,\"centroid_y\":null,"
           "\"min_x\":null,\"min_y\":null,\"max_x\":null,\"max_y\":null,"
           "\"area\":0,\"mean_margin\":0,\"ids\":[]}\\n");
  } else {
    double area = fmax(0.0, max_x - min_x) * fmax(0.0, max_y - min_y);
    printf("{\"tag_count\":%d,\"centroid_x\":%.6f,\"centroid_y\":%.6f,"
           "\"min_x\":%.6f,\"min_y\":%.6f,\"max_x\":%.6f,\"max_y\":%.6f,"
           "\"area\":%.6f,\"mean_margin\":%.6f,\"ids\":[",
           tag_count,
           sum_x / tag_count,
           sum_y / tag_count,
           min_x,
           min_y,
           max_x,
           max_y,
           area,
           sum_margin / tag_count);
    for (int i = 0; i < tag_count; ++i) {
      apriltag_detection_t* detection = NULL;
      zarray_get(detections, i, &detection);
      printf("%s%d", (i == 0) ? "" : ",", detection->id);
    }
    printf("],\"detections\":[");
    for (int i = 0; i < tag_count; ++i) {
      apriltag_detection_t* detection = NULL;
      zarray_get(detections, i, &detection);
      printf("%s{\"id\":%d,\"center\":[%.6f,%.6f],\"corners\":[",
             (i == 0) ? "" : ",",
             detection->id,
             detection->c[0],
             detection->c[1]);
      for (int j = 0; j < 4; ++j) {
        printf("%s[%.6f,%.6f]",
               (j == 0) ? "" : ",",
               detection->p[j][0],
               detection->p[j][1]);
      }
      printf("],\"decision_margin\":%.6f}", detection->decision_margin);
    }
    printf("]}\\n");
  }

  apriltag_detections_destroy(detections);
  apriltag_detector_destroy(detector);
  tag36h11_destroy(family);
  free(gray);
  pnm_destroy(pnm);
  return 0;
}
