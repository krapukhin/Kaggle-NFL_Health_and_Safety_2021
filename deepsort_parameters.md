# Deepsort parameters


- #### REID_CKPT: "input/yolov5-deepsort-pytorch/ckpt.t7"

## 1. [class Deepsort: \_\_init\_\_](https://github.com/ZQPei/deep_sort_pytorch/blob/8cfe2467a4b1f4421ebf9fcbc157921144ffe7cf/deep_sort/deep_sort.py#L15)
    
    metric = NearestNeighborDistanceMetric("cosine", 
                                           max_cosine_distance = max_dist, 
                                           nn_budget)
    self.tracker = Tracker(metric, 
                           max_iou_distance=max_iou_distance, 
                           max_age=max_age, 
                           n_init=n_init)

### 1.1. [NearestNeighborDistanceMetric](https://github.com/ZQPei/deep_sort_pytorch/blob/8cfe2467a4b1f4421ebf9fcbc157921144ffe7cf/deep_sort/sort/nn_matching.py#L99)
A nearest neighbor distance metric that, for each target, returns the closest distance to any sample that has been observed so far.

    metric = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
- `metric` : str. Either "euclidean" or "cosine".
- #### MAX_DIST: 0.2
    `matching_threshold`: float. The matching threshold. Samples with larger distance are considered an invalid match.
    

- ####  NN_BUDGET: 30
    `budget` : Optional[int]. If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.

### 1.2. [Tracker](https://github.com/ZQPei/deep_sort_pytorch/blob/8cfe2467a4b1f4421ebf9fcbc157921144ffe7cf/deep_sort/sort/tracker.py#L10)
This is the multi-target tracker

    Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

- `metric` : `nn_matching.NearestNeighborDistanceMetric`. A distance metric for measurement-to-track association
- #### MAX_IOU_DISTANCE: 0.9
    `def _match(self, detections):`

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
                iou_matching.iou_cost, 
                self.max_iou_distance, # <---
                self.tracks,
                detections, 
                iou_track_candidates, 
                unmatched_detections)
        
    [`min_cost_matching()`](https://github.com/nwojke/deep_sort/blob/280b8bdb255f223813ff4a8679f3e1321b08cdfc/deep_sort/linear_assignment.py#L11) - Solve linear assignment problem ( `max_iou_distance` : float. Gating threshold. Associations with cost larger than this value are disregarded.)
    

- #### MAX_AGE: 15
    `max_age` : int. Maximum number of missed misses before a track is deleted.

- #### N_INIT: 3
    `n_init` : int. Number of consecutive detections before the track is confirmed. The track state is set to `Deleted` if a miss occurs within the first `n_init` frames.

## 2. [class Deepsort: update](https://github.com/ZQPei/deep_sort_pytorch/blob/8cfe2467a4b1f4421ebf9fcbc157921144ffe7cf/deep_sort/deep_sort.py#L28)
    def update(self, bbox_xywh, confidences, ori_img):
            ...
            # generate detections
            detections = [Detection(bbox_tlwh[i], 
                                    conf, 
                                    features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

            # run on non-maximum supression
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            ...
### 2.1. [Detection](https://github.com/ZQPei/deep_sort_pytorch/blob/8cfe2467a4b1f4421ebf9fcbc157921144ffe7cf/deep_sort/sort/detection.py#L5)
This class represents a bounding box detection in a single image.
`class Detection(object)`. 
- #### MIN_CONFIDENCE: 0.2
    generate detections

        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]
    
    
    `tlwh` : array_like. Bounding box in format `(x, y, w, h)`.
    `confidence` : float. Detector confidence score.
    `feature` : array_like. A feature vector that describes the object contained in this image.

### 2.2. [non_max_suppression](https://github.com/ZQPei/deep_sort_pytorch/blob/8cfe2467a4b1f4421ebf9fcbc157921144ffe7cf/deep_sort/sort/preprocessing.py#L6)
Suppress overlapping detections. Original code from [URL](http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/) has been adapted to include confidence score.
`non_max_suppression(boxes, max_bbox_overlap, scores=None)`

- #### NMS_MAX_OVERLAP: 0.5
  
        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores) <---
        detections = [detections[i] for i in indices]

     

    `boxes` : ndarray. Array of ROIs (x, y, width, height).
    `nms_max_overlap` : float. ROIs that overlap more than this values are suppressed.
    `scores` : Optional[array_like]. Detector confidence score.












        
        
        
