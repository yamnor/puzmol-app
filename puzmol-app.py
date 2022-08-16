import threading

#import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

import av
import cv2

from PIL import Image

import torch
import numpy as np
import pandas as pd
import math

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors

import py3Dmol
from stmol import showmol

atomcolor = {
  'C' : {'min' : (  0,   0,   0), 'max' : (180, 255,  40)},
  'N' : {'min' : ( 90,  64, 128), 'max' : (150, 255, 255)},
  'O' : {'min' : (160,  64,   0), 'max' : (180, 255, 255)}}

bondtype = {
  'none'   : 0,
  'single' : 1,
  'double' : 2,
  'triple' : 3}

def draw_text(img, txt, xy):
  face = cv2.FONT_HERSHEY_PLAIN
  size = 2
  thickness = 3
  color = (255, 255, 255)
  (w, h), _ = cv2.getTextSize(txt, face, size, thickness)
  org = [int(xy[0] - w / 2), int(xy[1] + h / 2)]
  cv2.putText(img, txt, org, face, size, color, thickness, cv2.LINE_AA)

def draw_bond(img, atom, adjmat):
  geom = atom['geom']
  natoms = len(geom)
  colors = [
    (255, 255, 255), # none
    (  0, 255,   0), # single
    (  0,   0, 255), # double
    (255,   0,   0)] # triple
  for i in range(natoms):
    for j in range(i + 1, natoms):
      xi, yi = geom[i]
      xj, yj = geom[j]
      aij = int(adjmat[i, j])
      if aij > 0:
        cv2.line(img, (xi, yi), (xj, yj), colors[aij], thickness=2, lineType=cv2.LINE_AA)

def draw_atom(img, atom):
  natoms = len(atom['type'])
  for i in range(natoms):
    draw_text(img, atom['type'][i], atom['geom'][i])

def adj2mol(atom, adjmat):
  natoms = len(atom)
  mol = Chem.RWMol()
  idx = {}
  for i in range(natoms):
    idx[i] = mol.AddAtom(Chem.Atom(atom[i]))
  for i in range(natoms):
    for j in range(i + 1, natoms):
      aij = adjmat[i, j]
      if aij == 0:
        continue
      elif aij == 1:
        mol.AddBond(idx[i], idx[j], Chem.rdchem.BondType.SINGLE)
      elif aij == 2:
        mol.AddBond(idx[i], idx[j], Chem.rdchem.BondType.DOUBLE)
      elif aij == 2:
        mol.AddBond(idx[i], idx[j], Chem.rdchem.BondType.TRIPLE)
  return mol.GetMol()

def smi2mol(smi):
  mol = Chem.MolFromSmiles(smi)
  if mol is not None:
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, maxIters = 200)
    return mol
  else:
    return None

def img2smi(img, model):

  objects = {}
  for n in ['atom', 'bond']:
    objects[n] = model[n](img).pandas().xyxy[0]
  natoms = len(objects['atom'])
  nbonds = len(objects['bond'])

  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  atom = {'type' : [], 'geom' : []}
  for i in range(natoms):
    xmin = int(objects['atom'].xmin[i])
    ymin = int(objects['atom'].ymin[i])
    xmax = int(objects['atom'].xmax[i])
    ymax = int(objects['atom'].ymax[i])
    w = xmax - xmin
    h = ymax - ymin
    cx = int(xmin + w / 2)
    cy = int(ymin + h / 2)
    crop_xmin = int(xmin + w * 0.2)
    crop_ymin = int(ymin + h * 0.2)
    crop_xmax = int(xmax - w * 0.2)
    crop_ymax = int(ymax - h * 0.2)
    crop_w = crop_xmax - crop_xmin
    crop_h = crop_ymax - crop_ymin
    cropped = hsv[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    likelihood = {'C' : 0.0, 'N': 0.0, 'O' : 0.0}
    for a in ['C', 'N', 'O']:
      mask = cv2.inRange(cropped, atomcolor[a]['min'], atomcolor[a]['max'])
      likelihood[a] = sum(mask.flatten()) / 255 / (crop_w * crop_h)
    if max(likelihood.values()) > 0.5:
      name = max(likelihood, key=likelihood.get)
      atom['type'].append(name)
      atom['geom'].append(np.array([cx, cy]))
  natoms = len(atom['type'])

  bond = {'type' : [], 'geom' : []}
  for i in range(nbonds):
    xmin = int(objects['bond'].xmin[i])
    ymin = int(objects['bond'].ymin[i])
    xmax = int(objects['bond'].xmax[i])
    ymax = int(objects['bond'].ymax[i])
    cx = int((xmin + xmax) / 2)
    cy = int((ymin + ymax) / 2)
    bond['type'].append(objects['bond'].name[i])
    bond['geom'].append(np.array([cx, cy]))

  dismat = np.zeros((natoms, natoms))
  mindis = 1000.0
  for i in range(natoms):
    for j in range(i + 1, natoms):
      xi, yi = atom['geom'][i]
      xj, yj = atom['geom'][j]
      dismat[i, j] = math.sqrt((xi - xj)**2 + (yi - yj)**2)
      dismat[j, i] = dismat[i, j]
      if dismat[i, j] < mindis:
        mindis = dismat[i, j]

  adjmat = np.zeros((natoms, natoms))
  for i in range(natoms):
    for j in range(i + 1, natoms):
      xi, yi = atom['geom'][i]
      xj, yj = atom['geom'][j]
      cx = int((xi + xj) / 2)
      cy = int((yi + yj) / 2)
      if dismat[i, j] < mindis * 1.2:
        aij = 0
        for k in range(nbonds):
          xk, yk = bond['geom'][k]
          if (cx - mindis / 2) < xk < (cx + mindis / 2) and (cy - mindis / 2) < yk < (cy + mindis / 2):
            aij = bondtype[bond['type'][k]]
        adjmat[i, j] = aij
        adjmat[j, i] = aij

  draw_bond(img, atom, adjmat)
  draw_atom(img, atom)

  return Chem.MolToSmiles(adj2mol(atom['type'], adjmat))

def show_2dview(smi):
  mol = Chem.MolFromSmiles(smi)
  if mol is not None:
    col = st.columns(3)
    col[0].write(' ')
    col[1].image(Draw.MolToImage(mol))
    col[2].write(' ')
  else:
    st.error('Try again.')

def show_3dview(smi):
  viewsize = (400, 700)
  mol = smi2mol(smi)
  if mol is not None:
    viewer = py3Dmol.view(height = viewsize[0], width = viewsize[1])
    molblock = Chem.MolToMolBlock(mol)
    viewer.addModel(molblock, 'mol')
    viewer.setStyle({'stick':{}})
    viewer.zoomTo()
    viewer.spin('y', 1)
    showmol(viewer, height = viewsize[0], width = viewsize[1])
    st.balloons()
  else:
    st.error('Try again.')

def show_properties(smi):
  mol = Chem.MolFromSmiles(smi)
  if mol is not None:
    col = st.columns(4)
    col[0].metric(label = "Molelular Weight",    value = Descriptors.MolWt(mol))
    col[1].metric(label = "Hetero Atoms",        value = Descriptors.NumHeteroatoms(mol))
    col[2].metric(label = "sp3 Carbon Fraction", value = Descriptors.FractionCSP3(mol))
    col[3].metric(label = "Rotatable Bonds",     value = Descriptors.NumRotatableBonds(mol))
    col = st.columns(4)
    col[0].metric(label = "Log P",               value = Descriptors.MolLogP(mol))
    col[1].metric(label = "Polar Surface Area",  value = Descriptors.TPSA(mol))
    col[2].metric(label = "H-bond Acceptors",    value = Descriptors.NumHAcceptors(mol))
    col[3].metric(label = "H-bond Donors",       value = Descriptors.NumHDonors(mol))
  else:
    st.error('Try again.')

def main():

  st.set_page_config(
    page_title = 'PuzMol',
    #page_icon = 'logo2.png',
    initial_sidebar_state = 'auto')

  class VideoProcessor:
    frame_lock: threading.Lock
    smi: None

    def __init__(self) -> None:
      self.frame_lock = threading.Lock()
      self.smi = None
      self.model = {}
      self.model['atom'] = torch.hub.load('ultralytics/yolov5', 'custom', path = 'model/puzmol-atom.pt')
      self.model['bond'] = torch.hub.load('ultralytics/yolov5', 'custom', path = 'model/puzmol-bond.pt')

    def recv(self, frame):
      img = frame.to_ndarray(format="bgr24")
      smi = img2smi(img, self.model)
      with self.frame_lock:
        self.smi = smi
      return av.VideoFrame.from_ndarray(img, format="bgr24")

  with st.sidebar:

    st.title('About')

    st.markdown(
      """
      Paper-craft molecular model can be read with a camera
      to convert it into 2D & 3D structures and predict its basic chemical properties.

      * The **green**, **red**, and **blue** lines shown on the video screen represent
        **single**, **double**, and **triple** bonds, respectively.
      """)

    st.warning(
      """
      This web app is hosted on a cloud server ([Streamlit Cloud](https://streamlit.io/))
      and videos are sent to the server for processing.
      
      No data is stored, everything is processed in memory and discarded,
      but if this is a concern for you, please refrain from using this service.
      """
    )

    st.info(
      """
      This web app is maintained by [Norifumi Yamamoto (@yamnor)](https://twitter.com/yamnor).
      
      You can follow me on social media:
      [GitHub](https://github.com/yamnor) | 
      [LinkedIn](https://www.linkedin.com/in/yamnor) | 
      [WebSite](https://yamlab.net).
      """)

  st.title("PuzMol")

  #with st.expander("Webcam Live Feed", expanded = True):
  ctx = webrtc_streamer(
    key = "puzmol",
    media_stream_constraints = {"video": True, "audio": False},
    video_processor_factory = VideoProcessor,
    rtc_configuration = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

  st.markdown("---")

  if ctx.video_processor:

    st.markdown("Click **Smile!**, when the detected structure is OK.")

    if st.button("Smile!"):

      with ctx.video_processor.frame_lock:
        smi = ctx.video_processor.smi

        if smi is not None:          
          st.markdown("---")

          st.subheader('SMILES')
          st.code(smi)
          st.markdown(
            """
            [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system)
            is a specification for describing the chemical structure of molecules using short strings.
            """)

          st.markdown("---")

          st.subheader('2D View')
          show_2dview(smi)

          st.markdown("---")

          st.subheader('3D View')
          show_3dview(smi)

          st.markdown("---")

          st.subheader('Properties')
          show_properties(smi)

          st.markdown("---")
        else:
          st.warning("No frames available yet.")
    
if __name__ == "__main__":
    main()
