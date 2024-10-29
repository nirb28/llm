# Author: Daljeet Singh
# jupyter nbconvert --Exporter.preprocessors=common.preprocess.ExtractAttachmentsPreprocessor --to notebook demoes/llm-book/inference/inference.ipynb
# --inplace

from binascii import a2b_base64
import sys
import os

from traitlets import Unicode, Set
from nbconvert.preprocessors import Preprocessor

class ExtractAttachmentsPreprocessor(Preprocessor):
    """
    Extracts all of the outputs from the notebook file.  The extracted
    outputs are returned in the 'resources' dictionary.
    """

    output_filename_template = Unicode(
        "attach_{cell_index}_{name}"
    ).tag(config=True)

    extract_output_types = Set(
        {'image/png', 'image/jpeg', 'image/svg+xml', 'application/pdf'}
    ).tag(config=True)

    def preprocess_cell(self, cell, resources, cell_index):
        print("ExtractAttachmentsPreprocessor.preprocess_cell")
        """
        Apply a transformation on each cell,

        Parameters
        ----------
        cell : NotebookNode cell
            Notebook cell being processed
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.
        cell_index : int
            Index of the cell being processed (see base.py)
        """

        # Get files directory if it has been specified
        output_files_dir = resources.get('output_files_dir', None)

        # Make sure outputs key exists
        if not isinstance(resources['outputs'], dict):
            resources['outputs'] = {}

        # Loop through all of the attachments in the cell
        for name, attach in cell.get("attachments", {}).items():
            for mime, data in attach.items():
                if mime not in self.extract_output_types:
                    continue

                # Binary files are base64-encoded, SVG is already XML
                if mime in {'image/png', 'image/jpeg', 'application/pdf'}:
                    # data is b64-encoded as text (str, unicode),
                    # we want the original bytes
                    data = a2b_base64(data)
                elif sys.platform == 'win32':
                    data = data.replace('\n', '\r\n').encode("UTF-8")
                else:
                    data = data.encode("UTF-8")

                filename = self.output_filename_template.format(
                    cell_index=cell_index,
                    name=name,)

                if output_files_dir is not None:
                    filename = os.path.join(output_files_dir, filename)

                if name.endswith(".gif") and mime == "image/png":
                    filename = filename.replace(".gif", ".png")

                # In the resources, make the figure available via
                #   resources['outputs']['filename'] = data
                resources['outputs'][filename] = data

                # now we need to change the cell source so that it links to the
                # filename instead of `attachment:`
                attach_str = "attachment:"+name
                if attach_str in cell.source:
                    cell.source = cell.source.replace(attach_str, filename)

        return cell, resources
