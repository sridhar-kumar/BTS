import torch
import numpy as np
import cv2

def real_gradcam(model, tensor, original_image):
    """
    Real Grad-CAM implementation for U-Net encoder
    """

    model.eval()

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Hook last encoder block
    target_layer = model.unet.encoder.layer4

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)

    output = model(tensor)
    loss = output.mean()
    model.zero_grad()
    loss.backward()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze()

    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)

    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR),
        0.6,
        heatmap,
        0.4,
        0
    )

    handle_fw.remove()
    handle_bw.remove()

    return overlay