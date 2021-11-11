import cv2
import numpy as np
import torch
import ttach as tta
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection


class BaseCAM:
    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None,
                 compute_input_gradient=False,
                 uses_gradients=True, **kwargs):
        self.model = model.eval()       # Turn off gradient computations
        self.target_layers = target_layers      # The selected layer for gradCAM/CAM
        self.cuda = use_cuda        # Enable CUDA if required
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform      # If the selected layer is a non-convolution layer
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

        # In cases where the output is not straightforward
        # and is a dictionary
        if kwargs.keys():
            self._classification_logits = kwargs["classification_logit"]

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = [0.0]
        for i in range(len(target_category)):

            print(type(output))
            # Cases where we need to classify and also have a bounding box
            if isinstance(output, dict):

                # Format the output tensor -> In DeTR it is a bit different
                with torch.no_grad():

                    # Get the classes to check for multiple occurances
                    predictions = output[self._classification_logits].argmax(axis=2)
                    temp = (predictions == target_category[i]).nonzero(as_tuple=True)
                    # Stack them by rows
                    # Each row for a multi target_category in a single image
                    target_locs = torch.stack(temp, axis=1)
                    # Get the number of target categories
                    num_target_categories = target_locs.shape[0]

                    if num_target_categories > 1:
                        for j in range(num_target_categories-1):
                            loss.append(0.0)

                    #logits_bbox_id = output[self._classification_logits][i, :, target_category].argmax()

                for t_cat_inst in range(num_target_categories):
                    loss[t_cat_inst] = loss[t_cat_inst] + output[self._classification_logits][i, target_locs[t_cat_inst, 1],
                                                                      target_category[i]]

            elif isinstance(output, tuple):

                print(output)
                predictions = output[0]
                max_score_bb_index, max_class_bb = output[1]

                loss[0] = loss[0] + predictions[i, max_score_bb_index, 5+max_class_bb]

            else:
                loss[0] = loss[0] + output[i, target_category[i]]

        print(f"The max logit value is {loss}")

        # Make sure that the loss is a tensor
        if isinstance(loss, float):
            raise Exception("The target category is missing in the prediction")

        return loss[0]

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_layer,
                                       target_category, activations, grads)
        weighted_activations = weights[:, :, None, None] * activations

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor, target_category=None, eigen_smooth=False):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            # Cases where the prediction result is a dict - Classification and bounding boxes
            if isinstance(output, dict):
                target_category = np.argmax(output["pred_logits"].cpu().data.numpy(), axis=-1) # Wont work
            else:
                target_category = np.argmax(output.cpu().data.numpy(), axis=-1)

        else:
            assert(len(target_category) == input_tensor.size(0))

        if self.uses_gradients:
            self.model.zero_grad()
            loss = self.get_loss(output, target_category)
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   target_category,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor,
            target_category,
            eigen_smooth):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for target_layer, layer_activations, layer_grads in \
                zip(self.target_layers, activations_list, grads_list):
            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     target_category,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam[cam<0]=0 # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    def scale_cam_image(self, cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               target_category, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False):

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, target_category, eigen_smooth)

        return self.forward(input_tensor,
                            target_category, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
