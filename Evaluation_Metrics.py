import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score, precision_score, recall_score, accuracy_score

class Evaluation_Metrics:

    def F1_Score(self, masks, preds, mode):
        if mode == "segmentation":
            temp_masks = np.array(masks).flatten()
            temp_preds = np.array(preds).flatten()
        else:
            temp_masks = masks
            temp_preds = preds

        f1 = f1_score(temp_masks, temp_preds, average="weighted")
        f1_macro = f1_score(temp_masks, temp_preds, average="macro")
        f1_micro = f1_score(temp_masks, temp_preds, average="micro")
        print(f"f1 score weighted: {round(f1, 4)}")
        print(f"f1_macro score macro: {round(f1_macro, 4)}")
        print(f"f1_micro score macro: {round(f1_micro, 4)}")

        precision = precision_score(temp_masks, temp_preds, average="weighted")
        precision_macro = precision_score(temp_masks, temp_preds, average="macro")
        precision_micro = precision_score(temp_masks, temp_preds, average="micro")
        print(f"precision score weighted: {round(precision, 4)}")
        print(f"precision_macro score macro: {round(precision_macro, 4)}")
        print(f"precision_micro score micro: {round(precision_micro, 4)}")

        recall = recall_score(temp_masks, temp_preds, average="weighted")
        recall_macro = recall_score(temp_masks, temp_preds, average="macro")
        recall_micro = recall_score(temp_masks, temp_preds, average="micro")
        print(f"recall score weighted: {round(recall, 4)}")
        print(f"recall_macro score macro: {round(recall_macro, 4)}")
        print(f"recall_micro score micro: {round(recall_micro, 4)}")
        
        return f1

        # all_f1_score = []
        # for i in range(len(masks)):
        #     mask_flat = np.copy(masks[i]).flatten()
        #     pred_flat = np.copy(preds[i]).flatten()
        #     f1 = f1_score(mask_flat, pred_flat, zero_division = 0)
        #     all_f1_score.append(f1)
        #     # print(f"[{i}] {test_paths[i]} -----> {accuracy}")
        # print(f"mean F1_Score = {round(np.mean(all_f1_score), 4)} ± {round(np.std(all_f1_score), 4)}")
        # return all_f1_score

    def pixelwise_accuracy(self, masks, preds, mode):
        temp_masks = np.array(masks).flatten()
        temp_preds = np.array(preds).flatten()
        correct_pixels = np.sum(temp_masks == temp_preds)
        total_pixels = temp_masks.size
        accuracy = correct_pixels / total_pixels
        sklearn_accuracy_score = accuracy_score(temp_masks, temp_preds)
        print(f"pixelwise accuracy: {accuracy}")
        print(f"sklearn_accuracy_score: {sklearn_accuracy_score}")
        return accuracy

        # all_accuracy = []
        # for i in range(len(masks)):
        #     mask_flat = np.copy(masks[i]).flatten()
        #     pred_flat = np.copy(preds[i]).flatten()
        #     correct_pixels = np.sum(mask_flat == pred_flat)
        #     accuracy = correct_pixels/mask_flat.size
        #     all_accuracy.append(accuracy)
        #     # print(f"[{i}] {test_paths[i]} -----> {accuracy}")
        # print(f"mean Accuracy = {round(np.mean(all_accuracy), 4)} ± {round(np.std(all_accuracy), 4)}")
        # return all_accuracy
    
    def iou(self, masks, preds, mode):
        temp_masks = np.array(masks).flatten()
        temp_preds = np.array(preds).flatten()
        iou = jaccard_score(temp_masks, temp_preds, average="weighted")
        iou_macro = jaccard_score(temp_masks, temp_preds, average="macro")
        iou_micro = jaccard_score(temp_masks, temp_preds, average="micro")
        print(f"IoU: {iou}")
        print(f"iou_macro: {iou_macro}")
        print(f"iou_micro: {iou_micro}")
        return iou
        # intersection = np.logical_and(temp_masks, temp_preds).sum()
        # union = np.logical_or(temp_masks, temp_preds).sum()
        # if union == 0 and intersection == 0:
        #     iou = 1.0
        # else:
        #     iou = intersection/union
        # print(f"IoU: {iou}")
        # return iou

        # all_iou = []
        # for i in range(len(masks)):
        #     intersection = np.logical_and(masks[i], preds[i]).sum()
        #     union = np.logical_or(masks[i], preds[i]).sum()
        #     if union == 0 and intersection == 0:
        #         iou = 1.0
        #     else:
        #         iou = intersection/union
        #     all_iou.append(iou)
        #     # print(f"[{i}] {test_paths[i]} -----> {iou}")
        # print(f"mean IoU = {round(np.mean(all_iou), 4)} ± {round(np.std(all_iou), 4)}")
        # return all_iou
    
    def classification_accuracy(self, true_class_labels, pred_class_labels, mode):
        correct_preds = np.sum(np.array(true_class_labels) == np.array(pred_class_labels))
        accuracy = correct_preds / len(true_class_labels)
        print(f"classification accuracy: {accuracy}")
        return accuracy
    
        # all_accuracy = []
        # for i in range(len(true_class_labels)):
        #     true_class = true_class_labels[i]
        #     correctness = int(true_class == pred_class_labels[i])
        #     all_accuracy.append(correctness)
        # print(f"mean Accuracy = {round(np.mean(all_accuracy), 4)} ± {round(np.std(all_accuracy), 4)}")
        # return all_accuracy
    
    def make_classification_report(self, true_class_labels, pred_class_labels, mode):
        class_names = ["slum", "non-slum", "both"]
        labels = [0, 1, 2]
        class_report = classification_report(true_class_labels, pred_class_labels, target_names = class_names, labels = labels, zero_division = 0)
        print(class_report)
        return class_report
        
    def make_confusion_matrix(self, true_class_labels, pred_class_labels, mode):
        # class_names = ["slum", "non-slum", "both"]
        confusion_mtrx = confusion_matrix(true_class_labels, pred_class_labels)
        print(confusion_mtrx)
        return confusion_mtrx

    def group_labels(self, labels):
        slums = 0
        non_slums = 0
        mixed = 0
        for i in labels:
            if i == 0:
                slums += 1
            elif i == 1:
                non_slums += 1
            elif i == 2:
                mixed += 1
        print(f"slums: {slums}")
        print(f"non_slums: {non_slums}")
        print(f"mixed: {mixed}")
