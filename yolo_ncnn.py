import cv2
import numpy as np
import ncnn
from typing import List, Tuple

class YoloV5NCNN:
    def __init__(self, param_path: str, bin_path: str, input_size: Tuple[int, int] = (640, 640)):
        """
        Initialize YOLOv5 NCNN model
        
        Args:
            param_path: Path to the NCNN .param file
            bin_path: Path to the NCNN .bin file
            input_size: Input size as (width, height), default is 640x640
        """
        self.net = ncnn.Net()
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        self.input_width, self.input_height = input_size
        self.mean_vals = [0, 0, 0]
        self.norm_vals = [1/255.0, 1/255.0, 1/255.0]
        
        # Default parameters for YOLOv5
        self.conf_threshold = 0.25
        self.nms_threshold = 0.45
        self.class_names = self._get_class_names()

    def _get_class_names(self) -> List[str]:
        """Get COCO class names"""
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def detect(self, img_path: str) -> List[dict]:
        """
        Run object detection on an image
        
        Args:
            img_path: Path to the input image
            
        Returns:
            List of detection results, each containing:
            - 'class_id': class ID
            - 'label': class name
            - 'confidence': detection confidence
            - 'box': bounding box as [x, y, width, height]
        """
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        return self.detect_image(img)
    
    def detect_image(self, img: np.ndarray) -> List[dict]:
        """
        Run object detection on an image array
        
        Args:
            img: OpenCV image array (BGR format)
            
        Returns:
            List of detection results
        """
        img_height, img_width = img.shape[:2]
        
        # Create NCNN extractor
        ex = self.net.create_extractor()
        
        # Preprocess image
        # NCNN requires specific preprocessing steps
        img_resized = cv2.resize(img, (self.input_width, self.input_height))
        # Convert to RGB (from BGR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Create NCNN Mat directly
        mat_in = ncnn.Mat.from_pixels(img_rgb, ncnn.Mat.PixelType.PIXEL_RGB, 
                                      self.input_width, self.input_height)
        
        # Apply normalization
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
        
        # Set input
        ex.input("images", mat_in)
        
        # Extract output
        # YOLOv5 typically has one output layer
        out = ncnn.Mat()
        ret = ex.extract("output", out)  # YOLOv5-NCNN typically uses "output" as the output layer name
        
        if ret != 0:
            print(f"Failed to extract output, return code: {ret}")
            return []
        
        # Post-process detections
        results = []
        
        # Process YOLOv5 output
        # out shape is typically [num_boxes, 5 + num_classes]
        # where first 5 values are [x, y, w, h, obj_conf]
        rows = out.h  # number of detections
        
        # Extract detections
        detections = []
        for i in range(rows):
            row_data = []
            for j in range(out.w):
                row_data.append(out.channel(0)[i*out.w + j])
            detections.append(row_data)
        
        # Apply confidence threshold
        filtered_detections = []
        for detection in detections:
            # Get confidence and class scores
            confidence = detection[4]
            if confidence >= self.conf_threshold:
                classes_scores = detection[5:]
                class_id = np.argmax(classes_scores)
                class_score = classes_scores[class_id]
                
                # Combine objectness score with class confidence
                confidence = float(confidence * class_score)
                
                if confidence >= self.conf_threshold:
                    # Convert bbox to [x, y, w, h] format and scale to original image size
                    cx, cy = detection[0], detection[1]
                    w, h = detection[2], detection[3]
                    
                    # Calculate top-left corner
                    x = (cx - w/2) / self.input_width * img_width
                    y = (cy - h/2) / self.input_height * img_height
                    w = w / self.input_width * img_width
                    h = h / self.input_height * img_height
                    
                    filtered_detections.append({
                        'box': [x, y, w, h],
                        'confidence': confidence,
                        'class_id': int(class_id)
                    })
        
        # Apply NMS
        boxes = np.array([d['box'] for d in filtered_detections])
        confidences = np.array([d['confidence'] for d in filtered_detections])
        class_ids = np.array([d['class_id'] for d in filtered_detections])
        
        # Convert boxes to the format expected by NMS
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        boxes_for_nms = np.stack([x1, y1, x2, y2], axis=1)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes_for_nms.tolist(), confidences.tolist(), 
                                  self.conf_threshold, self.nms_threshold)
        
        # Prepare final results
        for i in indices:
            if isinstance(i, list):  # For OpenCV 3.x compatibility
                i = i[0]
            
            detection = filtered_detections[i]
            results.append({
                'class_id': detection['class_id'],
                'label': self.class_names[detection['class_id']],
                'confidence': float(detection['confidence']),
                'box': [float(v) for v in detection['box']]  # x, y, w, h
            })
            
        return results
    
    def draw_detections(self, img: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Draw detection results on the image
        
        Args:
            img: Original image
            detections: List of detection results
            
        Returns:
            Image with drawn detections
        """
        result_img = img.copy()
        
        for detection in detections:
            # Get box coordinates
            x, y, w, h = detection['box']
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Get class information
            class_id = detection['class_id']
            label = detection['label']
            confidence = detection['confidence']
            
            # Generate random color based on class id
            color = (int(hash(str(class_id)) % 255), 
                     int(hash(str(class_id*2)) % 255), 
                     int(hash(str(class_id*3)) % 255))
            
            # Draw box
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(result_img, label_text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return result_img
    
    def set_conf_threshold(self, threshold: float) -> None:
        """Set confidence threshold"""
        self.conf_threshold = threshold
    
    def set_nms_threshold(self, threshold: float) -> None:
        """Set NMS threshold"""
        self.nms_threshold = threshold

# Example usage
if __name__ == "__main__":
    # Initialize YOLOv5 NCNN detector
    detector = YoloV5NCNN(
        param_path="yolov5nu_ncnn_model/model.ncnn.param",
        bin_path="yolov5nu_ncnn_model/model.ncnn.bin"
    )
    
    # Run detection on image
    image_path = "photos/2/15.jpg"

    
    # Method 2: Using image array
    img = cv2.imread(image_path)
    img = cv2.flip(img, 0)
    img= cv2.resize(img, (640, 360))
    detections = detector.detect_image(img)
    
    # Draw detections
    result_img = detector.draw_detections(img, detections)

    print(detections)
    
    # Display results
    for detection in detections:
        print(f"Detected {detection['label']} with confidence {detection['confidence']:.2f}")
    
    # Save result
    cv2.imwrite("result.jpg", result_img)
    
    # Display (if running in environment with display)
    cv2.imshow("YOLOv5 NCNN Detection", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()