using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Unity.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class DepthImage : MonoBehaviour
{
    public Text info;
    public Text objectInfo;

    // Get or set the AROcclusionManager.
    public AROcclusionManager occlusionManager {
        get => m_OcclusionManager;
        set => m_OcclusionManager = value;
    }
    [SerializeField]
    [Tooltip("The AROcclusionManager which will produce depth textures.")]
    AROcclusionManager m_OcclusionManager;

    // Get or set the ARCameraManager.
    public ARCameraManager cameraManager {
        get => m_CameraManager;
        set => m_CameraManager = value;
    }
    [SerializeField]
    [Tooltip("The ARCameraManager which will produce camera frame events.")]
    ARCameraManager m_CameraManager;

    // Depth image
    public RawImage rawImage {
        get => m_RawImage;
        set => m_RawImage = value;
    }
    [SerializeField]
    RawImage m_RawImage;

    // Camera image
    public RawImage rawCameraImage {
        get => m_RawCameraImage;
        set => m_RawCameraImage = value;
    }
    [SerializeField]
    RawImage m_RawCameraImage;

    // This is for visualizing depth images
    public Material depthMaterial {
        get => m_DepthMaterial;
        set => m_DepthMaterial = value;
    }
    [SerializeField]
    Material m_DepthMaterial;

    // Using multiple audio sources to queue collision audio with no delay
    public AudioSource[] audioSources;
    private int audioSelect = 0; // Select audio source
    private double lastScheduled = -10; // mag = -1 for left, mag = 1 for right

    private double audioDuration; // Audio duration = 0.0853333333333333

    // Depth image data
    byte[] depthArray = new byte[0];
    int depthWidth = 0; // (width, height) = (256, 192) on iPhone 12 Pro
    int depthHeight = 0;
    int depthStride = 4; // Should be either 2 or 4
    
    // Camera intrinsics
    Vector2 focalLength = Vector2.zero;
    Vector2 principalPoint = Vector2.zero;

    private new Camera camera;

    public static Vector3 position;
    public static Vector3 rotation;

    private bool doObstacleAvoidance = false;

    public Toggle depthToggle;

    double curTime = 0;
    double lastDSP = 0;

    RunYOLO yolo;

    void Awake()
    {
        Application.targetFrameRate = 30;
        QualitySettings.vSyncCount = 0;

        camera = m_CameraManager.GetComponent<Camera>();

        m_CameraManager.frameReceived += OnCameraFrameReceived;

        // Set depth image material
        m_RawImage.material = m_DepthMaterial;

        m_RawImage.enabled = doObstacleAvoidance;

        audioDuration = (double) audioSources[0].clip.samples / audioSources[0].clip.frequency;

        yolo = GameObject.Find("ScriptHandler").GetComponent<RunYOLO>();
        Debug.unityLogger.Log("mytag", Screen.height);
        Debug.unityLogger.Log("mytag", Screen.width);
    }

    void OnCameraFrameReceived(ARCameraFrameEventArgs args)
    {
        UpdateDepthImages();
        UpdateCameraImage();
    }

    // This is called every frame
    void Update()
    {
        // Show the FPS
        info.text = $"{Convert.ToInt32(1.0 / Time.unscaledDeltaTime)}\n";

        // Update timer for collision audio
        if (lastDSP != AudioSettings.dspTime) {
            lastDSP = AudioSettings.dspTime;
            curTime = lastDSP;
        }
        else curTime += Time.unscaledDeltaTime;

        // Check if an object was detected
        // Assumptions: portrait mode; depth image height = RGB image height;
        //     depth image resolution = 480x640; model input/output resolution = 480x480
        if (RunYOLO.objectDetected) {
            // Convert object pixel location in the 480x480 camera image to a pixel location in the depth image
            int depthX = (int) (depthWidth / 2 + depthWidth / 640f * RunYOLO.objectPosition.y);
            int depthY = (int) (depthHeight / 2 - depthWidth / 640f * RunYOLO.objectPosition.x);

            // Get depth & position of object relative to camera
            float depth = GetDepth(depthX, depthY);
            RunYOLO.objectPos3d = ComputeVertex(depthX, depthY, depth);
            objectInfo.text = String.Format("{0}m, {1}°", RunYOLO.objectPos3d.magnitude.ToString("F2"),
                                                          Math.Atan2(RunYOLO.objectPos3d.x, RunYOLO.objectPos3d.z).ToString("F2"));

            // Play audio; fastest rate is 11Hz when next to object, slowest rate is 2Hz when at least 5m from object
            float rate = rate = Mathf.Lerp(10, 2, depth/5);
            PlayCollision(0, 1/rate - audioDuration);

            RunYOLO.objectDetected = false;
        }
    }

    private bool UpdateDepthImages()
    {
        bool success = false;

        // Acquire a depth image and update the corresponding raw image.
        if (m_OcclusionManager.TryAcquireEnvironmentDepthCpuImage(out XRCpuImage image)) {
            using (image) {
                UpdateRawImage(m_RawImage, image, image.format.AsTextureFormat(), true);

                // Get distance data into depthArray
                depthWidth = image.width;
                depthHeight = image.height;
                UpdateCameraParams();

                int numPixels = depthWidth * depthHeight;
                Debug.Assert(image.planeCount == 1, "Plane count is not 1");
                depthStride = image.GetPlane(0).pixelStride;
                int numBytes = numPixels * depthStride;
                if (depthArray.Length != numBytes)
                    depthArray = new byte[numBytes];
                image.GetPlane(0).data.CopyTo(depthArray);

                success = true;
            }
        }

        return success;
    }

    // Access RGB camera images
    private void UpdateCameraImage()
    {
        // Acquire a camera image, update the corresponding raw image, and do CV
        if (m_CameraManager.TryAcquireLatestCpuImage(out XRCpuImage cameraImage)) {
            using (cameraImage) {
                UpdateRawImage(m_RawCameraImage, cameraImage, TextureFormat.RGB24, false);
                StartCoroutine(yolo.ExecuteML(m_RawCameraImage.texture));
            }
        }
    }

    // Need to handle multiple audio sources so we can schedule sufficiently ahead of time, particularly at low FPS.
    private void PlayCollision(float dir, double delay)
    {
        float rad = (dir + rotation.y) * Mathf.Deg2Rad;
        this.transform.position = position + new Vector3(Mathf.Sin(rad), 0, Mathf.Cos(rad));

        double nextSchedule = Math.Max(curTime, lastScheduled + audioDuration + delay);
        while (nextSchedule - curTime < 0.2 && !audioSources[audioSelect].isPlaying) { // Schedule next audio if it will be needed soon
            audioSources[audioSelect].PlayScheduled(nextSchedule);
            audioSelect = (audioSelect + 1) % audioSources.Length;
            lastScheduled = nextSchedule;
            nextSchedule = Math.Max(curTime, lastScheduled + audioDuration + delay);
        }
    }

    public void ToggleObstacleAvoidance()
    {
        doObstacleAvoidance = depthToggle.isOn;
        m_RawImage.enabled = doObstacleAvoidance;
    }

    private void UpdateRawImage(RawImage rawImage, XRCpuImage cpuImage, TextureFormat format, bool isDepth)
    {
        Debug.Assert(rawImage != null, "no raw image");

        // Get the texture associated with the UI.RawImage that we wish to display on screen.
        var texture = rawImage.texture as Texture2D;

        // If the texture hasn't yet been created, or if its dimensions have changed, (re)create the texture.
        // Note: Although texture dimensions do not normally change frame-to-frame, they can change in response to
        //    a change in the camera resolution (for camera images) or changes to the quality of the human depth
        //    and human stencil buffers.
        if (texture == null || texture.width != cpuImage.width || texture.height != cpuImage.height)
        {
            texture = new Texture2D(cpuImage.width, cpuImage.height, format, false);
            rawImage.texture = texture;
        }

        // For display, we need to mirror about the vertical axis.
        var conversionParams = new XRCpuImage.ConversionParams(cpuImage, format, XRCpuImage.Transformation.MirrorY);

        // Get the Texture2D's underlying pixel buffer.
        var rawTextureData = texture.GetRawTextureData<byte>();

        // Make sure the destination buffer is large enough to hold the converted data (they should be the same size)
        Debug.Assert(rawTextureData.Length == cpuImage.GetConvertedDataSize(conversionParams.outputDimensions, conversionParams.outputFormat),
            "The Texture2D is not the same size as the converted data.");

        // Perform the conversion.
        cpuImage.Convert(conversionParams, rawTextureData);

        // "Apply" the new pixel data to the Texture2D.
        texture.Apply();

        // Get the aspect ratio for the current texture.
        var textureAspectRatio = (float)texture.width / texture.height;

        // Determine the raw image rectSize preserving the texture aspect ratio, matching the screen orientation,
        // and keeping a minimum dimension size.
        float minDimension = 480.0f;
        float maxDimension = Mathf.Round(minDimension * textureAspectRatio);
        Vector2 rectSize;
        if (isDepth) {
            maxDimension = Screen.height;
            minDimension = Screen.width;
            switch (Screen.orientation)
            {
                case ScreenOrientation.LandscapeRight:
                case ScreenOrientation.LandscapeLeft:
                    rectSize = new Vector2(maxDimension, minDimension);
                    break;
                case ScreenOrientation.PortraitUpsideDown:
                case ScreenOrientation.Portrait:
                default:
                    rectSize = new Vector2(minDimension, maxDimension);
                    break;
            }
            rawImage.rectTransform.sizeDelta = rectSize;

            // Rotate the depth material to match screen orientation.
            Quaternion rotation = Quaternion.Euler(0, 0, GetRotation());
            Matrix4x4 rotMatrix = Matrix4x4.Rotate(rotation);
            m_RawImage.material.SetMatrix(Shader.PropertyToID("_DisplayRotationPerFrame"), rotMatrix);
        }
        else {
            rectSize = new Vector2(maxDimension, minDimension);
            rawImage.rectTransform.sizeDelta = rectSize;
        }
    }

    // Obtain the depth value in meters. (x,y) are pixel coordinates
    // In portrait mode, (0, 0) is top right, (depthWidth, depthHeight) is bottom left.
    // Screen orientation does not change coordinate locations on the screen.
    public float GetDepth(int x, int y)
    {
        if (depthArray.Length == 0)
            return 99999f;

        /*
        Different phones may give image data in different formats
        */
        int index = (y * depthWidth) + x;
        float depthInMeters = 0;
        if (depthStride == 4) // DepthFloat32
            depthInMeters = BitConverter.ToSingle(depthArray, depthStride * index);
        else if (depthStride == 2) // DepthUInt16
            depthInMeters = BitConverter.ToUInt16(depthArray, depthStride * index) * 0.001f;

        if (depthInMeters > 0) {
            return depthInMeters;
        }

        return 99999f;
    }

    // Given image pixel coordinates (x,y) and distance z, returns a point in local camera space, i.e. where camera is at (0,0,0) and z-axis is forward from camera
    public Vector3 ComputeVertex(int x, int y, float z)
    {
        Vector3 vertex = Vector3.negativeInfinity;
        if (z > 0) {
            float vertex_x = (x - principalPoint.x) * z / focalLength.x;
            float vertex_y = (y - principalPoint.y) * z / focalLength.y;
            vertex.x = vertex_x;
            vertex.y = -vertex_y;
            vertex.z = z;
        }
        return vertex;
    }

    public static int GetRotation() => Screen.orientation switch
    {
        ScreenOrientation.Portrait => 90,
        ScreenOrientation.LandscapeLeft => 180,
        ScreenOrientation.PortraitUpsideDown => -90,
        ScreenOrientation.LandscapeRight => 0,
        _ => 90
    };

    private int GetRotationForScreen() => Screen.orientation switch
    {
        ScreenOrientation.Portrait => -90,
        ScreenOrientation.LandscapeLeft => 0,
        ScreenOrientation.PortraitUpsideDown => 90,
        ScreenOrientation.LandscapeRight => 180,
        _ => -90
    };

    private void UpdateCameraParams()
    {
        // Gets the camera parameters to create the required number of vertices.
        if (m_CameraManager.TryGetIntrinsics(out XRCameraIntrinsics cameraIntrinsics))
        {
            // Scales camera intrinsics to the depth map size.
            Vector2 intrinsicsScale;
            intrinsicsScale.x = depthWidth / (float)cameraIntrinsics.resolution.x;
            intrinsicsScale.y = depthHeight / (float)cameraIntrinsics.resolution.y;

            focalLength = MultiplyVector2(cameraIntrinsics.focalLength, intrinsicsScale);
            principalPoint = MultiplyVector2(cameraIntrinsics.principalPoint, intrinsicsScale);
            focalLength.y = focalLength.x;
        }
    }

    private Vector2 MultiplyVector2(Vector2 v1, Vector2 v2)
    {
        return new Vector2(v1.x * v2.x, v1.y * v2.y);
    }
}