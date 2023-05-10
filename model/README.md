Trying out an idea. Basically:

```mermaid
graph TD;
pc("Point Cloud [N x 3]");
dec_in("Embeddings [N x N_feat*(K+1)]");
Open3D("Open3D surface reconstruction (Screened Poisson)");

pc-->KDTree;
pc-->enc;
enc("PointNet++ Encoder")-->enc_out("Embeddings [N x N_feat]");
enc_out-->dec_in;
KDTree -- Embedding from K neighbors --> dec_in;
dec_in --> dec(Convolutional Decoder);
dec --> dec_out("Estimated Normals [N x 3]");

dec_out-->post;
pc-->post;

post("Post processing")-->Open3D;

```

> The background noise points are set to be predicted as all zeros (which is never possible for a real normal). The output will be post_processed to spot these points using a small threshold and remove them. Foreground noise points are currently staying in the output.
