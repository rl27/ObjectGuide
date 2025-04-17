// Reference: https://huggingface.co/unity/sentis-YOLOv8n

using System.Collections;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using FF = Unity.Sentis.Functional;


public class RunYOLO : MonoBehaviour
{
    [Tooltip("Drag a YOLO model .onnx file here")]
    public ModelAsset modelAsset;

    [Tooltip("Drag the classes.txt here")]
    public TextAsset classesAsset;

    [Tooltip("Create a Raw Image in the scene and link it here")]
    public RawImage displayImage;

    [Tooltip("Drag a border box texture here")]
    public Texture2D borderTexture;

    [Tooltip("Select an appropriate font for the labels")]
    public Font font;

    BackendType backend;

    private Transform displayLocation;
    private Worker worker;
    private string[] labels;
    private Sprite borderSprite;

    // Image size for the model
    private const int imageWidth = 480;
    private const int imageHeight = 480;

    List<GameObject> boxPool = new();

    [Tooltip("Intersection over union threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)] float iouThreshold = 0.5f;

    [Tooltip("Confidence score threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)] float scoreThreshold = 0.5f;

    Tensor<float> centersToCorners;
    public struct BoundingBox
    {
        public float centerX; // center = 0, right = positive
        public float centerY; // center = 0, down = positive
        public float width;
        public float height;
        public string label;
    }

    bool testing = false;
    Texture2D testPNG;

    TextureResizer resizer;
    TextureResizer.ResizeOptions resizeOptions;

    float displayWidth;
    float displayHeight;

    private const string objectClass = "cup";
    public static bool objectDetected = false;
    public static Vector2 objectPosition = Vector2.zero; // Pixel location of object center
    public static Vector3 objectPos3d = Vector3.zero; // Location of object relative to camera

    void Start()
    {
        displayWidth = displayImage.rectTransform.rect.width;
        displayHeight = displayImage.rectTransform.rect.height;

        backend = SystemInfo.supportsComputeShaders ? BackendType.GPUCompute : BackendType.CPU;

        // Test object detection in Unity editor
        #if UNITY_EDITOR
            testing = true;
            testPNG = VisionUtils.LoadPNG("Assets/Scripts/test.png");
            displayImage.enabled = true;
        #else
            // Scale the locations of bounding boxes to be accurate on phone display
            displayWidth = Screen.height * 480 / 640;
            displayHeight = Screen.height * 480 / 640;
        #endif

        // Parse class labels
        labels = classesAsset.text.Split('\n');

        LoadModel();

        displayLocation = displayImage.transform;

        borderSprite = Sprite.Create(borderTexture, new Rect(0, 0, borderTexture.width, borderTexture.height), new Vector2(borderTexture.width / 2, borderTexture.height / 2));

        // This is for resizing the input image
        resizer = new TextureResizer();
        resizeOptions = new TextureResizer.ResizeOptions()
        {
            aspectMode = TextureResizer.AspectMode.Fill,
            rotationDegree = 90,
            mirrorHorizontal = false,
            mirrorVertical = false,
            width = imageWidth,
            height = imageHeight,
        };

        // Do this to deal with initial lag
        // using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, Texture2D.blackTexture.height, Texture2D.blackTexture.width));
        // TextureConverter.ToTensor(Texture2D.blackTexture, inputTensor, default);
    }

    // Resize image using specified resizeOptions
    private Texture2D ResizeTexture(Texture inputTex)
    {
        resizeOptions.rotationDegree = DepthImage.GetRotation();
        RenderTexture resizedTex = resizer.Resize(inputTex, resizeOptions);
        return RenderTo2D(resizedTex);
    }

    Texture2D tex2D;
    private Texture2D RenderTo2D(RenderTexture texture)
    {
        if (tex2D == null || tex2D.width != texture.width || tex2D.height != texture.height)
            tex2D = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, false);
        var prevRT = RenderTexture.active;
        RenderTexture.active = texture;

        tex2D.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0);
        tex2D.Apply();

        RenderTexture.active = prevRT;

        return tex2D;
    }

    // Load model from file
    void LoadModel()
    {
        var model1 = ModelLoader.Load(modelAsset);

        centersToCorners = new Tensor<float>(new TensorShape(4, 4),
        new float[]
        {
                    1,      0,      1,      0,
                    0,      1,      0,      1,
                    -0.5f,  0,      0.5f,   0,
                    0,      -0.5f,  0,      0.5f
        });

        // Here we transform the output of the model1 by feeding it through a Non-Max-Suppression layer.
        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model1);
        var modelOutput = FF.Forward(model1, inputs)[0];                        //shape=(1,84,8400)
        var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);               //shape=(8400,4)
        var allScores = modelOutput[0, 4.., ..];                                //shape=(80,8400)
        var scores = FF.ReduceMax(allScores, 0);                                //shape=(8400)
        var classIDs = FF.ArgMax(allScores, 0);                                 //shape=(8400)
        var boxCorners = FF.MatMul(boxCoords, FF.Constant(centersToCorners));   //shape=(8400,4)
        var indices = FF.NMS(boxCorners, scores, iouThreshold, scoreThreshold); //shape=(N)
        var coords = FF.IndexSelect(boxCoords, 0, indices);                     //shape=(N,4)
        var labelIDs = FF.IndexSelect(classIDs, 0, indices);                    //shape=(N)

        // Create worker to run model
        worker = new Worker(graph.Compile(coords, labelIDs), backend);
    }

    private void Update()
    {
        if (testing) {
            StartCoroutine(ExecuteML(testPNG));
        }
    }

    bool working = false;
    int layersPerFrame = 80;
    public IEnumerator ExecuteML(Texture inputTex)
    {
        // Exit function if already running model
        if (working) yield break;
        working = true;

        Texture2D resizedTex = ResizeTexture(inputTex);
        // using Tensor<float> inputTensor = TextureConverter.ToTensor(resizedTex);
        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));
        TextureConverter.ToTensor(resizedTex, inputTensor, default);
        worker.Schedule(inputTensor);

        // Execute model over multiple frames
        var enumerator = worker.ScheduleIterable(inputTensor); 
        int it = 0;
        while (enumerator.MoveNext()) {
            if (++it % layersPerFrame == 0)
                yield return null;
        }

        yield return null;
        using var output = (worker.PeekOutput("output_0") as Tensor<float>).ReadbackAndClone();
        using var labelIDs = (worker.PeekOutput("output_1") as Tensor<int>).ReadbackAndClone();
        yield return null;

        float scaleX = displayWidth / imageWidth;
        float scaleY = displayHeight / imageHeight;

        #if UNITY_EDITOR // Show image if in Unity editor
            displayImage.texture = resizedTex;
        #endif

        // Redraw the bounding boxes
        ClearAnnotations();
        int boxesFound = output.shape[0];
        objectDetected = false;
        for (int n = 0; n < Mathf.Min(boxesFound, 50); n++)
        {
            var box = new BoundingBox
            {
                centerX = output[n, 0] * scaleX - displayWidth / 2,
                centerY = output[n, 1] * scaleY - displayHeight / 2,
                width = output[n, 2] * scaleX,
                height = output[n, 3] * scaleY,
                label = labels[labelIDs[n]],
            };
            DrawBox(box, n, displayHeight * 0.05f);

            if (box.label == objectClass) {
                objectPosition = new Vector2(box.centerX, box.centerY);
                objectDetected = true;
            }
        }

        working = false;
    }

    public void DrawBox(BoundingBox box, int id, float fontSize)
    {
        //Create the bounding box graphic or get from pool
        GameObject panel;
        if (id < boxPool.Count)
        {
            panel = boxPool[id];
            panel.SetActive(true);
        }
        else
        {
            panel = CreateNewBox(Color.yellow);
        }
        //Set box position
        panel.transform.localPosition = new Vector3(box.centerX, -box.centerY);

        //Set box size
        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(box.width, box.height);

        //Set label text
        var label = panel.GetComponentInChildren<Text>();
        label.text = box.label;
        label.fontSize = (int)fontSize;
    }

    public GameObject CreateNewBox(Color color)
    {
        // Create the box and set image
        var panel = new GameObject("ObjectBox");
        panel.AddComponent<CanvasRenderer>();
        Image img = panel.AddComponent<Image>();
        img.color = color;
        img.sprite = borderSprite;
        img.type = Image.Type.Sliced;
        panel.transform.SetParent(displayLocation, false);

        // Create the label
        var text = new GameObject("ObjectLabel");
        text.AddComponent<CanvasRenderer>();
        text.transform.SetParent(panel.transform, false);
        Text txt = text.AddComponent<Text>();
        txt.font = font;
        txt.color = color;
        txt.fontSize = 40;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;

        RectTransform rt2 = text.GetComponent<RectTransform>();
        rt2.offsetMin = new Vector2(20, rt2.offsetMin.y);
        rt2.offsetMax = new Vector2(0, rt2.offsetMax.y);
        rt2.offsetMin = new Vector2(rt2.offsetMin.x, 0);
        rt2.offsetMax = new Vector2(rt2.offsetMax.x, 30);
        rt2.anchorMin = new Vector2(0, 0);
        rt2.anchorMax = new Vector2(1, 1);

        boxPool.Add(panel);
        return panel;
    }

    public void ClearAnnotations()
    {
        foreach (var box in boxPool)
        {
            box.SetActive(false);
        }
    }

    private void OnDestroy()
    {
        centersToCorners?.Dispose();
        worker?.Dispose();
    }
}
