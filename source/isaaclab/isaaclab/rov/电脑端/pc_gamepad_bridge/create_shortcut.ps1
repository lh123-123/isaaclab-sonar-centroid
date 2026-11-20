# Render a vector-style ROV icon and create a local launcher shortcut.
# Does not launch QGC, connect to the vehicle, or create a desktop shortcut.
$ErrorActionPreference = 'Stop'
$rovProject = Split-Path -Parent $PSScriptRoot
$rovBatName = -join ([char[]](0x6BD4,0x8D5B,0x542F,0x52A8))
$rovTarget = Join-Path $rovProject ($rovBatName + '.bat')
$rovShortcut = Join-Path $rovProject ($rovBatName + '.lnk')
$rovIcon = Join-Path $PSScriptRoot 'rov.ico'
$rovPreview = Join-Path $PSScriptRoot 'rov_icon_preview.png'
if (-not (Test-Path -LiteralPath $rovTarget -PathType Leaf)) { throw 'Launcher BAT not found.' }
$rovShell = New-Object -ComObject WScript.Shell
if (Test-Path -LiteralPath $rovShortcut) {
    $rovExisting = $rovShell.CreateShortcut($rovShortcut)
    if ($rovExisting.TargetPath -ne $rovTarget) { throw 'Existing shortcut targets another file; not overwritten.' }
}
Add-Type -AssemblyName System.Drawing
Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @'
using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
public static class RovLauncherArtwork {
    static Bitmap Render(int size) {
        var bitmap = new Bitmap(size, size, PixelFormat.Format32bppArgb);
        using (var g = Graphics.FromImage(bitmap)) {
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.ScaleTransform(size / 256f, size / 256f);
            using (var background = new SolidBrush(Color.FromArgb(11, 25, 44)))
            using (var hull = new SolidBrush(Color.FromArgb(29, 64, 87)))
            using (var accent = new Pen(Color.FromArgb(73, 217, 237), 9))
            using (var prop = new SolidBrush(Color.FromArgb(73, 217, 237)))
            using (var white = new SolidBrush(Color.FromArgb(236, 250, 255))) {
                g.FillEllipse(background, 2, 2, 252, 252);
                var outline = new PointF[] {new PointF(85,50),new PointF(171,50),
                    new PointF(191,74),new PointF(191,179),new PointF(171,203),
                    new PointF(85,203),new PointF(65,179),new PointF(65,74)};
                g.FillPolygon(hull, outline);
                g.DrawPolygon(accent, outline);
                g.FillPolygon(white, new PointF[]{new PointF(128,21),new PointF(115,40),new PointF(141,40)});
                float[,] hubs = {{59,65},{197,65},{59,190},{197,190},{94,102},{162,102},{94,151},{162,151}};
                for (int i=0; i<8; i++) {
                    float x=hubs[i,0], y=hubs[i,1];
                    g.FillEllipse(background,x-18,y-18,36,36);
                    g.DrawEllipse(accent,x-17,y-17,34,34);
                    g.FillEllipse(prop,x-6,y-6,12,12);
                }
                // Camera/light bar, deliberately legible at small icon sizes.
                g.FillRectangle(white,108,116,40,22);
                g.FillEllipse(background,120,120,16,16);
                g.DrawLine(accent,112,209,112,225);
                g.DrawLine(accent,144,209,144,225);
                g.DrawLine(accent,112,225,122,234);
                g.DrawLine(accent,144,225,134,234);
            }
        }
        return bitmap;
    }
    public static void Save(string iconPath, string previewPath) {
        int[] sizes = {16,24,32,48,64,128,256};
        byte[][] images = new byte[sizes.Length][];
        for(int i=0;i<sizes.Length;i++) {
            using(var bitmap=Render(sizes[i])) using(var stream=new MemoryStream()) {
                bitmap.Save(stream,ImageFormat.Png);
                images[i]=stream.ToArray();
            }
        }
        using(var stream=File.Create(iconPath)) using(var writer=new BinaryWriter(stream)) {
            writer.Write((ushort)0); writer.Write((ushort)1); writer.Write((ushort)sizes.Length);
            uint offset=(uint)(6+16*sizes.Length);
            for(int i=0;i<sizes.Length;i++) {
                writer.Write((byte)(sizes[i]==256?0:sizes[i]));
                writer.Write((byte)(sizes[i]==256?0:sizes[i]));
                writer.Write((byte)0); writer.Write((byte)0);
                writer.Write((ushort)1); writer.Write((ushort)32);
                writer.Write((uint)images[i].Length); writer.Write(offset);
                offset+=(uint)images[i].Length;
            }
            foreach(var bytes in images) writer.Write(bytes);
        }
        using(var bitmap=Render(256)) bitmap.Save(previewPath,ImageFormat.Png);
    }
}
'@
[RovLauncherArtwork]::Save($rovIcon, $rovPreview)
$rovLink = $rovShell.CreateShortcut($rovShortcut)
$rovLink.TargetPath = $rovTarget
$rovLink.WorkingDirectory = $rovProject
$rovLink.IconLocation = $rovIcon + ',0'
$rovLink.Description = 'ROV competition launcher - QGC and controller monitor'
$rovLink.WindowStyle = 7
$rovLink.Save()
$rovCheck = $rovShell.CreateShortcut($rovShortcut)
if ($rovCheck.TargetPath -ne $rovTarget -or $rovCheck.IconLocation -ne ($rovIcon + ',0')) {
    throw 'Shortcut verification failed.'
}
[pscustomobject]@{ Shortcut=$rovShortcut; Target=$rovCheck.TargetPath; Icon=$rovCheck.IconLocation }
