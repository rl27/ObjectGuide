using UnityEngine;
using UnityEngine.UI;
using System.IO;

public class VisionUtils
{
    public static Texture2D LoadPNG(string filePath) {

        Texture2D tex = null;
        byte[] fileData;

        if (File.Exists(filePath)) {
            // Debug.Log("file loaded");
            fileData = File.ReadAllBytes(filePath);
            tex = new Texture2D(2, 2);
            tex.LoadImage(fileData);
        }
        return tex;
    }

    public static AspectRatioFitter.AspectMode GetMode() => Screen.orientation switch
    {
        ScreenOrientation.Portrait => AspectRatioFitter.AspectMode.WidthControlsHeight,
        ScreenOrientation.LandscapeLeft => AspectRatioFitter.AspectMode.HeightControlsWidth,
        ScreenOrientation.PortraitUpsideDown => AspectRatioFitter.AspectMode.WidthControlsHeight,
        ScreenOrientation.LandscapeRight => AspectRatioFitter.AspectMode.HeightControlsWidth,
        _ => AspectRatioFitter.AspectMode.WidthControlsHeight
    };
}
